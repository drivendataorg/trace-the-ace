"""Live Sandpiper annotation of test transcripts with Qwen2.5-7B-Instruct (``model_g_opt``).

Computes, inside the submission runtime container, the Sandpiper LLM features of the
``model_g_opt`` kind of ``models/submission/main.py``, in four annotation modes:

  tutor_whole  (every tutor utterance, session-wide)  -> 29 of the 36 LLM_SESS_COLS
  student      (every student utterance, session-wide) -> 7 of the 36 LLM_SESS_COLS
  tutor_rubric (per-session NTO quality rubric, 6 dims) -> LLM_RUBRIC_COLS
  tutor_lo_arc_misconception (per session-LO pair, confusion arc + misconception)
                                                        -> LLM_ARC_COLS (12, incl. PCA embed)

``models/submission/main.py`` imports this module
only for its ``model_g_opt`` kind. The training pipeline of this
release reuses :func:`build_prompts_whole` and :func:`_make_sampling_params` for the
interview-codebook annotation (``scripts/annotate_doc29_labels.py``), and
:mod:`magnificat.taxonomy_v2_prompts` reads three student definitions from
``STUDENT_SYSTEM_PROMPT``.

Prompts, schemas, parsing and feature formulas are copied as close to verbatim as possible
from the code that annotated the training set, so that training and inference features agree.

The vLLM settings the offline runtime needs are explained in
:func:`annotate_and_build_gopt_llm_matrix`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tutor taxonomy (prompt adapted from Sandpiper, MIT; see THIRD_PARTY_NOTICES.md)
# ---------------------------------------------------------------------------
TUTOR_CODES = [
    "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
    "PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP", "PROMPTING_SELF_CORRECTION",
    "FEEDBACK_CORRECT", "FEEDBACK_INCORRECT", "FEEDBACK_NEUTRAL",
    "REVOICING", "RESTATING", "GIVING_HINT", "GIVING_EXAMPLE",
    "EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
    "ADMINISTRATIVE",
]
TUTOR_SET = set(TUTOR_CODES)
SUBSTANTIVE_CODES = [c for c in TUTOR_CODES if c != "ADMINISTRATIVE"]

TUTOR_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "utterance_id": {"type": "integer"},
            "code": {"type": "string", "enum": TUTOR_CODES},
        },
        "required": ["utterance_id", "code"],
        "additionalProperties": False,
    },
}

TUTOR_SYSTEM_PROMPT = """\
You are an expert educational discourse analyst. Classify tutor utterances \
in a 1:1 maths tutoring session using the LEARNING_SUPPORT taxonomy below. \
Return ONLY a JSON array — no outer object, no explanation.

Allowed codes (use exactly one per utterance):
PROMPTING_RELATED_CONCEPTS, PROMPTING_ALTERNATIVE_REPRESENTATION,
PROMPTING_SELF_EXPLANATION, PROMPTING_NEXT_STEP, PROMPTING_SELF_CORRECTION,
FEEDBACK_CORRECT, FEEDBACK_INCORRECT, FEEDBACK_NEUTRAL,
REVOICING, RESTATING, GIVING_HINT, GIVING_EXAMPLE,
EXPLAINING_CONCEPTUAL, EXPLAINING_PROCEDURAL, GIVING_ANSWER,
ADMINISTRATIVE

Definitions:
- PROMPTING_SELF_EXPLANATION: asks student to justify reasoning ("How did you get that?")
- PROMPTING_NEXT_STEP: prompts the very next step without new info ("What do you do next?")
- PROMPTING_SELF_CORRECTION: flags error, asks student to find and fix it
- PROMPTING_RELATED_CONCEPTS: prompts connections to related ideas (no explanation given)
- PROMPTING_ALTERNATIVE_REPRESENTATION: asks student to represent idea differently
- FEEDBACK_CORRECT: explicitly confirms student answer is right ("Exactly!", "Yes")
- FEEDBACK_INCORRECT: explicitly states student answer is wrong ("Not quite")
- FEEDBACK_NEUTRAL: acknowledges without confirming correctness ("I see", "OK")
- GIVING_HINT: partial info guiding toward answer, does not reveal it
- GIVING_ANSWER: tutor directly provides the solution
- EXPLAINING_PROCEDURAL: step-by-step how-to guide
- EXPLAINING_CONCEPTUAL: explains the underlying concept or "why"
- REVOICING: repeats student's words with added precision or technical terms
- RESTATING: repeats student's words almost verbatim
- GIVING_EXAMPLE: uses an example different from the student's problem
- ADMINISTRATIVE: no pedagogical content — connection checks ("Hello?", "Can you hear me?"), \
unintelligible ([unclear]/[inaudible]), pure logistics ("Please wait", "One moment"), \
or content-free filler ("Yeah.", "Mm-hmm.") with no evaluative intent toward the student's learning

Output format: [{"utterance_id": <int>, "code": "<CODE>"}, ...]"""

# ---------------------------------------------------------------------------
# Student taxonomy
# ---------------------------------------------------------------------------
STUDENT_CODES = [
    "DEMONSTRATING_UNDERSTANDING",
    "CONFUSION_EXPRESSED",
    "SEEKING_CLARIFICATION",
    "SELF_CORRECTING",
    "OTHER",
    "ADMINISTRATIVE",
]
STUDENT_SET = set(STUDENT_CODES)

STUDENT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "utterance_id": {"type": "integer"},
            "code": {"type": "string", "enum": STUDENT_CODES},
        },
        "required": ["utterance_id", "code"],
        "additionalProperties": False,
    },
}

STUDENT_SYSTEM_PROMPT = """\
You are an expert educational discourse analyst. Classify student utterances \
in a 1:1 maths tutoring session according to their knowledge state. \
Return ONLY a JSON array — no outer object, no explanation.

Allowed codes (use exactly one per utterance):
DEMONSTRATING_UNDERSTANDING, CONFUSION_EXPRESSED, SEEKING_CLARIFICATION, \
SELF_CORRECTING, OTHER, ADMINISTRATIVE

Definitions:
- DEMONSTRATING_UNDERSTANDING: student produces correct reasoning, an answer (right or wrong \
  but substantive), or an explanation that reveals their thinking \
  ("So that means...", "I think it's X because...", "2 times 8 is 16, times 5 is 80")
- CONFUSION_EXPRESSED: student explicitly signals not-knowing or produces an error with no \
  self-correction ("I don't know", "I'm confused", "I'm not sure", gives a clearly wrong answer)
- SEEKING_CLARIFICATION: student asks a question about the material or task \
  ("Can you explain that?", "Did you say dividing by 10?", "What do you mean by...?")
- SELF_CORRECTING: student notices and fixes their own error in the SAME turn, without tutor \
  prompting ("Wait, no — it should be...", "Oh, I did it again... 24", "Actually, it's..."). \
  Do NOT use for subsequent correct answers after a tutor-guided re-attempt — those are \
  DEMONSTRATING_UNDERSTANDING.
- ADMINISTRATIVE: no cognitive engagement with the mathematics — backchannels ("Yeah", "Okay", \
  "Mm-hmm", "Uh-huh", "Right"), greetings/closings ("Hello", "Bye", "Thank you"), social talk \
  ("I'm good", "It was a good break"), reading slide or problem text aloud verbatim without \
  elaboration, connection/logistics talk ("Can you hear me?", "I'll try again"), or \
  unintelligible/[unclear] utterances
- OTHER: student utterance has some pedagogical relevance but does not fit the above — \
  rare residual category (metacognitive asides, emotional reactions to content, etc.)

Key distinctions:
- DEMONSTRATING_UNDERSTANDING covers substantive attempts even when wrong, and also includes \
  evaluating or disagreeing with someone else's reasoning ("I don't agree because...", \
  "That's the factor method not partitioning") — any turn where the student shows they are \
  thinking about the mathematics.
- CONFUSION_EXPRESSED is for explicit uncertainty ("I'm not sure", "I don't get it") or a \
  clearly wrong short answer with no reasoning shown. Mid-problem working steps are \
  DEMONSTRATING_UNDERSTANDING even when the student makes a mistake along the way.
- ADMINISTRATIVE vs OTHER: reading a question or slide aloud is ADMINISTRATIVE (passive); \
  adding any original comment or reaction to what was read makes it OTHER or \
  DEMONSTRATING_UNDERSTANDING.

Output format: [{"utterance_id": <int>, "code": "<CODE>"}, ...]"""

# ---------------------------------------------------------------------------
# NTO quality rubric (Sandpiper, MIT,
# github.com/National-Tutoring-Observatory/sandpiper)
# ---------------------------------------------------------------------------
RUBRIC_KEYS = [
    "reacting_to_errors",
    "giving_praise",
    "determining_what_students_know",
    "affirming_correct_attempt",
    "asking_guiding_questions",
    "prompting_explanation",
]

RUBRIC_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "boolean"} for k in RUBRIC_KEYS},
    "required": RUBRIC_KEYS,
    "additionalProperties": False,
}

RUBRIC_SYSTEM_PROMPT = """\
You are an expert educational discourse analyst. Analyze the 1:1 maths tutoring transcript below \
and score the tutor on 6 binary dimensions from the NTO Sandpiper quality rubric. \
Return ONLY a JSON object — no explanation.

Definitions:
- reacting_to_errors: true if the tutor responded to a student math error by asking them to \
explain their thinking OR prompting them to think again. false if the tutor gave the answer, \
pointed out the error directly, or no math error occurred.
- giving_praise: true if the tutor praised the student for their effort during the session. \
false if no praise given.
- determining_what_students_know: true if the tutor asked open-ended questions to check what \
the student already knows before or during a problem. false if no attempt to check prior knowledge.
- affirming_correct_attempt: true if the tutor explicitly affirmed a specific correct student \
response or reasoning by referencing what the student said or did. Generic affirmations \
('good', 'well done', 'exactly', 'great') without specifying what was correct score false. \
false if the tutor did not respond to a correct answer or reasoning attempt.
- asking_guiding_questions: true if the tutor asked at least one question requiring a \
substantive mathematical response — not yes/no, not a tag question ('right?', 'okay?', \
'yes?') — designed to guide the student toward discovering the answer themselves. Questions \
that only check recall or verify a known answer score false. false if the tutor only gave \
instructions or answers without guiding questions.
- prompting_explanation: true if the tutor prompted the student to explain their thinking. \
false if no prompt to explain thinking.

EXAMPLES for the two most commonly mis-scored dimensions:

affirming_correct_attempt — TRUE (tutor references the specific answer/step):
  S: "So x equals 5 because I subtracted 3 from both sides."
  T: "Yes — subtracting 3 from both sides is exactly the right move, and x = 5 is correct."

affirming_correct_attempt — FALSE (generic praise, no specifics):
  S: "I think it's 5."
  T: "Well done!"  [or]  T: "Exactly!"  [or]  T: "Great, carry on."

asking_guiding_questions — TRUE (requires substantive mathematical response):
  T: "What do you need to do to both sides to get x by itself?"
  T: "Why do we multiply numerator and denominator by the same number?"

asking_guiding_questions — FALSE (yes/no or tag question):
  T: "Does that make sense?"  [or]  T: "Is that right?"  [or]  T: "Okay?"

Output format: {"reacting_to_errors": true/false, "giving_praise": true/false, \
"determining_what_students_know": true/false, "affirming_correct_attempt": true/false, \
"asking_guiding_questions": true/false, "prompting_explanation": true/false}"""

# ---------------------------------------------------------------------------
# Confusion arc / misconception
# Methodology: Blank et al. (2026), arxiv.org/abs/2602.02414
# ---------------------------------------------------------------------------
ARC_MISCONCEPTION_KEYS = [
    "confusion_count", "resolution_count",
    "final_state", "resolution_agent",
    "misconception_present", "misconception_description",
    "misconception_resolved", "resolution_type",
]

ARC_MISCONCEPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "confusion_count":           {"type": "integer", "minimum": 0},
        "resolution_count":          {"type": "integer", "minimum": 0},
        "final_state":               {"type": "string",
                                      "enum": ["resolved", "partial", "unresolved", "no_confusion"]},
        "resolution_agent":          {"type": "string",
                                      "enum": ["student_led", "tutor_led", "collaborative", "na"]},
        "misconception_present":     {"type": "boolean"},
        "misconception_description": {"type": "string"},
        "misconception_resolved":    {"type": "boolean"},
        "resolution_type":           {"type": "string",
                                      "enum": ["student_self", "tutor_corrected", "partial", "none"]},
    },
    "required": ARC_MISCONCEPTION_KEYS,
    "additionalProperties": False,
}

ARC_MISCONCEPTION_SYSTEM_TEMPLATE = """\
You are an expert educational discourse analyst. You will be shown a dialogue excerpt from a \
1:1 maths tutoring session covering the learning objective below. Analyze the \
confusion-resolution arc and any student misconceptions. Return ONLY a JSON object with \
exactly 8 fields — no explanation.

Learning objective: {lo_text}

FIELDS:
- confusion_count (integer ≥ 0): distinct moments where the student expressed confusion, \
gave a wrong answer, or made a mathematical error in this excerpt.
- resolution_count (integer ≥ 0): of those confusion/error moments, how many were followed \
by the student demonstrating understanding (must be ≤ confusion_count; 0 if confusion_count=0).
- final_state: state at END of excerpt — "resolved" (student demonstrates understanding), \
"partial" (some progress, not fully resolved), "unresolved" (confusion present, not addressed), \
"no_confusion" (no confusion or errors detected).
- resolution_agent: who drove resolution — "student_led" (student self-corrected or figured \
it out), "tutor_led" (tutor gave the answer or full explanation), "collaborative" (joint \
effort), "na" (final_state is "no_confusion" or "unresolved").
- misconception_present (boolean): true ONLY if the student holds a specific identifiable \
incorrect mathematical belief — not mere confusion, uncertainty, or a calculation slip. \
Default to false when uncertain.
- misconception_description (string): if misconception_present is true, describe the specific \
false belief in ≤15 words (e.g. "student believes area equals perimeter for rectangles"). \
If false, use empty string "".
- misconception_resolved (boolean): if misconception_present is true, whether the student \
shows corrected understanding by end of excerpt. If misconception_present is false, use false.
- resolution_type (string): if misconception_present is true — "student_self" (student \
self-corrected), "tutor_corrected" (tutor explained/corrected), "partial" (partial progress), \
"none" (not resolved). If misconception_present is false, use "none".

Output format: {"confusion_count": <int>, "resolution_count": <int>, \
"final_state": "<string>", "resolution_agent": "<string>", \
"misconception_present": <bool>, "misconception_description": "<string>", \
"misconception_resolved": <bool>, "resolution_type": "<string>"}"""

# Fraction of arc/misconception prompts that may fail to parse before we abort; this
# catches a broken guided-decoding setup instead of shipping empty features.
SMOKE_TEST_FAILURE_THRESHOLD = 0.20

# ---------------------------------------------------------------------------
# Output column lists
# ---------------------------------------------------------------------------
LLM_SANDPIPER_COLS = [
    *[f"llm_{c.lower()}" for c in TUTOR_CODES],
    "llm_high_engagement_ratio", "llm_elicit_vs_tell", "llm_scaffold_vs_tell",
    "llm_feedback_positive_rate", "llm_move_entropy", "llm_giving_answer_rate",
]
LLM_TEMPORAL_COLS = [
    "llm_he_early", "llm_he_mid", "llm_he_late",
    "llm_scaffold_slope",
    "llm_fallback_count", "llm_fallback_rate",
    "llm_giving_answer_late",
]
LLM_STUDENT_COLS = [
    *[f"llm_stu_{c.lower()}" for c in STUDENT_CODES],
    "llm_stu_confusion_resolve",
    "llm_stu_late_demonstrating",
    "llm_stu_late_confusion",
    "llm_stu_learning_trajectory",
]
LLM_RUBRIC_COLS = [f"llm_rubric_{k}" for k in RUBRIC_KEYS]
LLM_ARC_COLS = [
    "llm_arc_confusion_count", "llm_arc_resolution_rate", "llm_arc_final_resolved",
    "llm_arc_student_led", "llm_misconception_present", "llm_misconception_resolved",
    "llm_misconception_student_self", "llm_misconception_tutor_corrected",
    "llm_misconception_embed_0", "llm_misconception_embed_1",
    "llm_misconception_embed_2", "llm_misconception_embed_3",
]

# The session columns: all of LLM_SANDPIPER_COLS + LLM_TEMPORAL_COLS, and 7 of
# LLM_STUDENT_COLS (not the 3 temporal-trajectory ones).
LLM_SESS_COLS = (
    LLM_SANDPIPER_COLS + LLM_TEMPORAL_COLS
    + [f"llm_stu_{c.lower()}" for c in STUDENT_CODES] + ["llm_stu_confusion_resolve"]
)


# ---------------------------------------------------------------------------
# LO-relevant window selection (only arc/misconception needs it: the window
# itself is not a feature, only the context selection for the arc/misconception
# prompt)
# ---------------------------------------------------------------------------
# Select the turns of one role most similar to the learning objective.
#
# Args:
#     df: one session's transcript.
#     lo_text: the learning-objective text.
#     role: "tutor" or "student".
#     encoder: a SentenceTransformer; turn and objective embeddings are normalised.
#     emb_cache: dict caching turn embeddings per (session_id, role) and objective
#         embeddings per text; updated in place.
#     session_id: the session's id (cache key).
#     top_k: number of turns to keep at most.
#     min_turns: number of turns to keep at least (if the session has them).
#
# Returns:
#     The selected rows of the role's turns, in transcript order.
def _lo_relevant_turns_semantic(
    df: pd.DataFrame, lo_text: str, role: str, encoder, emb_cache: dict,
    session_id: str, top_k: int = 20, min_turns: int = 3,
) -> pd.DataFrame:
    role_df = df[df["role"] == role].copy().reset_index(drop=True)
    n = len(role_df)
    if n == 0:
        return role_df

    cache_key = (session_id, role)
    if cache_key not in emb_cache:
        texts = role_df["content"].fillna("").tolist()
        emb_cache[cache_key] = encoder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )
    turn_embs = emb_cache[cache_key]

    lo_cache_key = ("__lo_emb__", lo_text)
    if lo_cache_key not in emb_cache:
        emb_cache[lo_cache_key] = encoder.encode(
            [lo_text], normalize_embeddings=True, show_progress_bar=False
        )[0]
    lo_emb = emb_cache[lo_cache_key]
    sims = turn_embs @ lo_emb

    k = max(min_turns, min(top_k, n))
    top_indices = sorted(np.argsort(sims)[-k:])
    return role_df.iloc[top_indices]


# Drop the longest turns until the prompt fits a character budget.
#
# Args:
#     turns: turn dicts with "utterance_id" and "content".
#     system_prompt: the system prompt (its length counts against the budget).
#     max_tokens: token budget for the prompt.
#     chars_per_token: characters assumed per token.
#
# Returns:
#     The remaining turns, in their original order (possibly empty).
def _truncate_to_fit(turns: list[dict], system_prompt: str, max_tokens: int = 3500,
                      chars_per_token: int = 4) -> list[dict]:
    overhead = len(system_prompt) + 100
    max_content_chars = max_tokens * chars_per_token - overhead
    turns = list(turns)
    while turns:
        chars = sum(len(f"utterance_id={t['utterance_id']}: {t.get('content', '')}") for t in turns)
        if chars <= max_content_chars:
            break
        i = max(range(len(turns)), key=lambda i: len(turns[i].get("content", "")))
        turns.pop(i)
    return turns


# Format one batch of utterances as a chat prompt for per-utterance classification.
#
# Args:
#     turns: turn dicts with "utterance_id" and "content".
#     system_prompt: the system prompt of the pass.
#     tokenizer: a Hugging Face tokenizer with a chat template.
#
# Returns:
#     The prompt string, with the generation prompt appended.
def _make_prompt(turns: list[dict], system_prompt: str, tokenizer) -> str:
    lines = [f"utterance_id={t['utterance_id']}: {t['content']}" for t in turns]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Classify these utterances:\n\n" + "\n".join(lines)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _parse_output(text: str, expected_ids: list[int]) -> list[dict]:
    """Parse a model reply into per-utterance code records.

    Accepts a JSON array, an object wrapping an array, a single object, or an
    ``{utterance_id: code}`` mapping.

    Args:
        text: the generated text.
        expected_ids: utterance ids of the batch, used when an item has no usable id.

    Returns:
        List of ``{"utterance_id": int, "code": str}``; empty if the text is not valid JSON.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        list_vals = [(k, v) for k, v in parsed.items() if isinstance(v, list)]
        if list_vals:
            items = list_vals[0][1]
        elif "utterance_id" in parsed or "code" in parsed:
            items = [parsed]
        else:
            items = []
            for k, v in parsed.items():
                if isinstance(v, dict) and "code" in v:
                    items.append(v)
                elif isinstance(v, str):
                    try:
                        items.append({"utterance_id": int(k), "code": v})
                    except ValueError:
                        pass
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    result = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            code = item.get("code") or item.get("label") or ""
            uid = item.get("utterance_id")
            if uid is None:
                uid = expected_ids[idx] if idx < len(expected_ids) else -1
            try:
                uid = int(uid)
            except (TypeError, ValueError):
                uid = expected_ids[idx] if idx < len(expected_ids) else -1
            result.append({"utterance_id": uid, "code": str(code)})
        elif isinstance(item, str):
            uid = expected_ids[idx] if idx < len(expected_ids) else -1
            result.append({"utterance_id": uid, "code": item})
    return result


# ---------------------------------------------------------------------------
# Prompt builders (tutor_whole / student share build_prompts_whole; rubric and
# arc_misconception each have their own)
# ---------------------------------------------------------------------------
# Build per-utterance classification prompts over every utterance of one role.
#
# Each session's utterances of the role are split into batches of at most batch_size
# utterances and at most max_prompt_tokens (estimated as characters / chars_per_token).
#
# Args:
#     session_ids: sessions to annotate; sessions that fail to load are skipped.
#     store: transcript store with a load(session_id) method.
#     batch_size: maximum utterances per prompt.
#     tokenizer: a Hugging Face tokenizer with a chat template.
#     role: "tutor" (annotation type "tutor_whole") or "student".
#     system_prompt: the system prompt of the pass.
#     max_prompt_tokens: token budget per prompt.
#     chars_per_token: characters assumed per token.
#
# Returns:
#     Tuple (meta_list, prompt_texts), aligned. Each meta dict holds session_id, lo_id (None),
#     annotation_type, expected_ids and position_fracs (within-role position in [0, 1]).
def build_prompts_whole(session_ids: list[str], store, batch_size: int, tokenizer,
                         role: str, system_prompt: str, max_prompt_tokens: int = 6500,
                         chars_per_token: int = 4) -> tuple[list[dict], list[str]]:
    # max_prompt_tokens=6500: max_model_len=8192 must cover prompt and
    # generation combined, and tutor_whole/student generate up to
    # max_tokens=1024 (see _make_sampling_params); 6500+1024=7524 leaves
    # headroom. In practice batch_size=40 is usually the binding constraint,
    # not this token budget.
    overhead_chars = len(system_prompt) + 200
    max_content_chars = max_prompt_tokens * chars_per_token - overhead_chars

    meta_list, prompt_texts = [], []
    for sid in session_ids:
        try:
            df = store.load(sid)
        except Exception:
            continue

        role_df = df[df["role"] == role][["utterance_id", "content"]]
        turns = role_df.to_dict("records")
        n_total = len(turns)
        if n_total == 0:
            continue

        ann_type = "tutor_whole" if role == "tutor" else "student"
        current_batch: list[dict] = []
        current_chars = 0
        base_idx = 0

        def _emit(batch: list[dict], base: int) -> None:
            expected_ids = [t["utterance_id"] for t in batch]
            pos_fracs = [(base + j) / max(n_total - 1, 1) for j in range(len(batch))]
            meta_list.append({
                "session_id": sid, "lo_id": None, "annotation_type": ann_type,
                "expected_ids": expected_ids, "position_fracs": pos_fracs,
            })
            prompt_texts.append(_make_prompt(batch, system_prompt, tokenizer))

        for turn in turns:
            turn_chars = len(f"utterance_id={turn['utterance_id']}: {turn.get('content', '')}")
            over_token_budget = current_chars + turn_chars > max_content_chars
            over_batch_limit = len(current_batch) >= batch_size
            if current_batch and (over_token_budget or over_batch_limit):
                _emit(current_batch, base_idx)
                base_idx += len(current_batch)
                current_batch = []
                current_chars = 0
            current_batch.append(turn)
            current_chars += turn_chars

        if current_batch:
            _emit(current_batch, base_idx)

    return meta_list, prompt_texts


# Build one rubric-scoring prompt per session from its tutor and student turns.
#
# Args:
#     session_ids: sessions to score; sessions that fail to load or have no turns are skipped.
#     store: transcript store with a load(session_id) method.
#     tokenizer: a Hugging Face tokenizer with a chat template.
#     max_prompt_tokens: token budget; the longest turns are dropped until it fits.
#
# Returns:
#     Tuple (meta_list, prompt_texts), aligned; each meta dict holds session_id.
def build_prompts_rubric(session_ids: list[str], store, tokenizer,
                          max_prompt_tokens: int = 3500) -> tuple[list[dict], list[str]]:
    meta_list, prompt_texts = [], []
    for sid in session_ids:
        try:
            df = store.load(sid)
        except Exception:
            continue

        df_filtered = df[df["role"].isin(["tutor", "student"])].copy()
        if df_filtered.empty:
            continue

        turns = df_filtered[["utterance_id", "role", "content"]].to_dict("records")
        turns = _truncate_to_fit(turns, RUBRIC_SYSTEM_PROMPT, max_prompt_tokens)
        if not turns:
            continue

        lines = [f"utterance_id={t['utterance_id']} [{t['role']}]: {t['content']}" for t in turns]
        messages = [
            {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
            {"role": "user", "content": "Transcript:\n\n" + "\n".join(lines) + "\n\nScore the 6 rubric dimensions:"},
        ]
        meta_list.append({"session_id": sid})
        prompt_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return meta_list, prompt_texts


# Build one confusion-arc / misconception prompt per (session, learning objective) pair.
#
# The excerpt is the span of tutor and student turns between the first and last of the
# top_k tutor turns most similar to the objective (_lo_relevant_turns_semantic).
#
# Args:
#     session_lo_pairs: (session_id, lo_id, lo_text) tuples.
#     store: transcript store with a load(session_id) method.
#     tokenizer: a Hugging Face tokenizer with a chat template.
#     encoder: SentenceTransformer used to select the objective-relevant turns.
#     emb_cache: embedding cache shared with _lo_relevant_turns_semantic; updated in place.
#     top_k: number of objective-relevant tutor turns used to set the span.
#     max_prompt_tokens: token budget; the longest turns are dropped until it fits.
#     df_cache: optional {session_id: transcript or None} cache; updated in place.
#
# Returns:
#     Tuple (meta_list, prompt_texts), aligned; each meta dict holds session_id and lo_id.
def build_prompts_arc_misconception(
    session_lo_pairs: list[tuple[str, str, str]], store, tokenizer, encoder, emb_cache: dict,
    top_k: int = 20, max_prompt_tokens: int = 4000, df_cache: dict | None = None,
) -> tuple[list[dict], list[str]]:
    if df_cache is None:
        df_cache = {}
    meta_list, prompt_texts = [], []

    for sid, lo_id, lo_text in session_lo_pairs:
        if sid not in df_cache:
            try:
                df_cache[sid] = store.load(sid)
            except Exception:
                df_cache[sid] = None
        df = df_cache[sid]
        if df is None:
            continue

        lo_df = _lo_relevant_turns_semantic(df, lo_text, "tutor", encoder, emb_cache, sid, top_k=top_k)
        if lo_df.empty:
            continue

        min_uid = lo_df["utterance_id"].min()
        max_uid = lo_df["utterance_id"].max()
        span_df = df[
            (df["utterance_id"] >= min_uid) & (df["utterance_id"] <= max_uid)
            & (df["role"].isin(["tutor", "student"]))
        ].copy().sort_values("utterance_id")
        if span_df.empty:
            continue

        system_prompt = ARC_MISCONCEPTION_SYSTEM_TEMPLATE.replace("{lo_text}", lo_text)
        turns = span_df[["utterance_id", "role", "content"]].to_dict("records")
        turns = _truncate_to_fit(turns, system_prompt, max_prompt_tokens)
        if not turns:
            continue

        lines = [f"utterance_id={t['utterance_id']} [{t['role']}]: {t['content']}" for t in turns]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Dialogue excerpt:\n\n" + "\n".join(lines) + "\n\nAnalyze the arc and misconception:"},
        ]
        meta_list.append({"session_id": sid, "lo_id": lo_id})
        prompt_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return meta_list, prompt_texts


# ---------------------------------------------------------------------------
# vLLM inference runners (no retry or resume code: this is a single-pass live job,
# not a resumable batch job)
# ---------------------------------------------------------------------------
def _make_sampling_params(schema: dict, max_tokens: int = 1024):
    """Return greedy vLLM sampling parameters, with JSON-schema constrained decoding unless SO_OFF=1.

    SO_OFF=1 is required for Qwen3.5-397B: its vocabulary merges `{` and `"` into one token
    (id 4754), and every vLLM grammar backend (xgrammar, guidance and outlines) builds its JSON
    FSM assuming those are separate symbols, so the FSM cannot advance past a valid JSON
    opening. The failure raises no error, so callers must count parse failures.

    Args:
        schema: JSON schema for the reply.
        max_tokens: maximum generated tokens.

    Returns:
        A ``vllm.SamplingParams`` with temperature 0.
    """
    import os  # noqa: PLC0415

    from vllm import SamplingParams
    if os.environ.get("SO_OFF") == "1":
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)
    from vllm.sampling_params import StructuredOutputsParams
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                           structured_outputs=StructuredOutputsParams(json=schema))


# Generate and parse per-utterance codes for a list of prompts.
#
# Args:
#     meta_list: prompt metadata from build_prompts_whole.
#     prompt_texts: the prompts, aligned with meta_list.
#     llm: a vllm.LLM.
#     sampling_params: from _make_sampling_params.
#     allowed_set: valid codes; other codes are dropped.
#     chunk_size: prompts per llm.generate call.
#     label: name of the pass (unused; nothing is printed).
#
# Returns:
#     List of records with session_id, utterance_id, code, annotation_type, lo_id and
#     position_frac. Unparseable replies are skipped without a count.
def run_vllm(meta_list: list[dict], prompt_texts: list[str], llm, sampling_params,
             allowed_set: set[str], chunk_size: int = 1000, label: str = "") -> list[dict]:
    n_total = len(prompt_texts)
    if n_total == 0:
        return []
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    records = []
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * chunk_size
        hi = min(lo + chunk_size, n_total)
        chunk_meta, chunk_prompts = meta_list[lo:hi], prompt_texts[lo:hi]
        outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)
        for meta, output in zip(chunk_meta, outputs):
            text = output.outputs[0].text
            annotations = _parse_output(text, meta["expected_ids"])
            if not annotations:
                continue
            pos_fracs = meta.get("position_fracs", [])
            for j, ann in enumerate(annotations):
                code = ann.get("code", "")
                if code not in allowed_set:
                    continue
                records.append({
                    "session_id": meta["session_id"], "utterance_id": ann["utterance_id"],
                    "code": code, "annotation_type": meta["annotation_type"],
                    "lo_id": meta["lo_id"],
                    "position_frac": pos_fracs[j] if j < len(pos_fracs) else float("nan"),
                })
    # No print here: n_total and len(records) are counts derived from the test
    # set, which the competition rules forbid logging.
    return records


def run_vllm_rubric(meta_list: list[dict], prompt_texts: list[str], llm, sampling_params,
                     chunk_size: int = 1000, label: str = "tutor_rubric") -> list[dict]:
    """Generate and parse the six binary rubric scores per session.

    Args:
        meta_list: prompt metadata from :func:`build_prompts_rubric`.
        prompt_texts: the prompts, aligned with ``meta_list``.
        llm: a ``vllm.LLM``.
        sampling_params: from :func:`_make_sampling_params`.
        chunk_size: prompts per ``llm.generate`` call.
        label: name of the pass (unused; nothing is printed).

    Returns:
        List of ``{"session_id", <rubric key>: 0 or 1, ...}``; replies that do not parse or
        miss a key are skipped.
    """
    n_total = len(prompt_texts)
    if n_total == 0:
        return []
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    records, n_failed = [], 0
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * chunk_size
        hi = min(lo + chunk_size, n_total)
        chunk_meta, chunk_prompts = meta_list[lo:hi], prompt_texts[lo:hi]
        outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)
        for meta, output in zip(chunk_meta, outputs):
            text = output.outputs[0].text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                n_failed += 1
                continue
            if not isinstance(parsed, dict) or not all(k in parsed for k in RUBRIC_KEYS):
                n_failed += 1
                continue
            rec = {"session_id": meta["session_id"]}
            valid = True
            for k in RUBRIC_KEYS:
                v = parsed[k]
                if not isinstance(v, (bool, int)):
                    valid = False
                    break
                rec[k] = int(bool(v))
            if not valid:
                n_failed += 1
                continue
            records.append(rec)
    # No print here, for the same reason as in run_vllm.
    return records


def run_vllm_arc_misconception(meta_list: list[dict], prompt_texts: list[str], llm,
                                sampling_params, chunk_size: int = 1000,
                                label: str = "tutor_lo_arc_misconception") -> list[dict]:
    """Generate and parse the confusion-arc / misconception record per (session, objective).

    Raises if the failure rate is implausibly high; this guard catches a broken
    guided-decoding setup in a smoke test, instead of silently producing empty features.

    Args:
        meta_list: prompt metadata from :func:`build_prompts_arc_misconception`.
        prompt_texts: the prompts, aligned with ``meta_list``.
        llm: a ``vllm.LLM``.
        sampling_params: from :func:`_make_sampling_params`.
        chunk_size: prompts per ``llm.generate`` call.
        label: name of the pass, used in the error message.

    Returns:
        List of records with session_id, lo_id and the eight ARC_MISCONCEPTION_KEYS fields
        (``misconception_resolved`` is None when no misconception is present).

    Raises:
        RuntimeError: if more than ``SMOKE_TEST_FAILURE_THRESHOLD`` of the replies fail to
            parse.
    """
    n_total = len(prompt_texts)
    if n_total == 0:
        return []
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    records, n_failed = [], 0
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * chunk_size
        hi = min(lo + chunk_size, n_total)
        chunk_meta, chunk_prompts = meta_list[lo:hi], prompt_texts[lo:hi]
        outputs = llm.generate(chunk_prompts, sampling_params, use_tqdm=False)
        for meta, output in zip(chunk_meta, outputs):
            text = output.outputs[0].text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                n_failed += 1
                continue
            if not isinstance(parsed, dict):
                n_failed += 1
                continue
            try:
                misconception_present = bool(parsed["misconception_present"])
                rec = {
                    "session_id": meta["session_id"], "lo_id": meta["lo_id"],
                    "confusion_count": int(parsed["confusion_count"]),
                    "resolution_count": int(parsed["resolution_count"]),
                    "final_state": str(parsed["final_state"]),
                    "resolution_agent": str(parsed["resolution_agent"]),
                    "misconception_present": misconception_present,
                    "misconception_description": str(parsed.get("misconception_description", "")),
                    "misconception_resolved": (
                        bool(parsed.get("misconception_resolved", False)) if misconception_present else None
                    ),
                    "resolution_type": str(parsed.get("resolution_type", "none")),
                }
                if not misconception_present:
                    rec["misconception_description"] = ""
                    rec["resolution_type"] = "none"
            except (KeyError, TypeError, ValueError):
                n_failed += 1
                continue
            records.append(rec)

    # No print here: the failure_rate below is used only for the abort decision and
    # is never logged. The exception message contains no test-derived counts.
    failure_rate = n_failed / n_total if n_total > 0 else 0.0
    if failure_rate > SMOKE_TEST_FAILURE_THRESHOLD:
        raise RuntimeError(
            f"{label}: JSON parse failure rate exceeded the "
            f"{SMOKE_TEST_FAILURE_THRESHOLD:.0%} threshold — aborting rather than "
            "silently shipping empty features (likely a vLLM/schema version mismatch)."
        )
    return records


# ---------------------------------------------------------------------------
# Feature compilation (formulas as in the training-set feature build)
# ---------------------------------------------------------------------------
# num / denom as a float, or `default` when denom is not positive.
def _safe_frac(num, denom, default=0.0):
    return float(num / denom) if denom > 0 else default


# Shannon entropy in bits of a count vector (0.0 for an all-zero vector).
def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# Boolean masks for the early (< 0.33), middle and late (>= 0.67) thirds of a session.
def _third_mask(pos: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    return pos < 0.33, (pos >= 0.33) & (pos < 0.67), pos >= 0.67


# Count PROMPTING_* codes immediately followed by GIVING_ANSWER (prompt, then give the answer).
def _count_fallbacks(codes: list[str]) -> int:
    n = 0
    for i in range(len(codes) - 1):
        if codes[i].startswith("PROMPTING_") and codes[i + 1] == "GIVING_ANSWER":
            n += 1
    return n


# Compute one session's tutor features from its tutor_whole annotations.
#
# Code rates use substantive (non-ADMINISTRATIVE) turns as denominator, except
# llm_administrative, which uses all turns. Thirds use position_frac (NaN -> 0.5).
#
# Args:
#     grp: annotation records of one session (utterance_id, code, position_frac).
#
# Returns:
#     Dict with the LLM_SANDPIPER_COLS and LLM_TEMPORAL_COLS values.
def _tutor_whole_features(grp: pd.DataFrame) -> dict:
    grp = grp.sort_values("utterance_id")
    total = len(grp)
    code_counts = grp["code"].value_counts()
    n_admin = code_counts.get("ADMINISTRATIVE", 0)
    n_substantive = total - n_admin

    subst_grp = grp[grp["code"] != "ADMINISTRATIVE"]
    pos = subst_grp["position_frac"].fillna(0.5)

    row: dict = {}
    for code in TUTOR_CODES:
        denom = total if code == "ADMINISTRATIVE" else n_substantive
        row[f"llm_{code.lower()}"] = _safe_frac(code_counts.get(code, 0), denom)

    prompting_total = sum(code_counts.get(c, 0) for c in SUBSTANTIVE_CODES if c.startswith("PROMPTING_"))
    explaining_total = code_counts.get("EXPLAINING_CONCEPTUAL", 0) + code_counts.get("EXPLAINING_PROCEDURAL", 0)
    giving_answer = code_counts.get("GIVING_ANSWER", 0)
    giving_hint = code_counts.get("GIVING_HINT", 0)
    giving_example = code_counts.get("GIVING_EXAMPLE", 0)
    fb_correct = code_counts.get("FEEDBACK_CORRECT", 0)
    fb_incorrect = code_counts.get("FEEDBACK_INCORRECT", 0)

    row["llm_high_engagement_ratio"] = _safe_frac(prompting_total, n_substantive)
    row["llm_elicit_vs_tell"] = _safe_frac(prompting_total, explaining_total + giving_answer + 1)
    row["llm_scaffold_vs_tell"] = _safe_frac(giving_hint + giving_example, giving_answer + 1)
    row["llm_feedback_positive_rate"] = _safe_frac(fb_correct, fb_correct + fb_incorrect)
    row["llm_giving_answer_rate"] = _safe_frac(giving_answer, n_substantive)
    counts_arr = np.array([code_counts.get(c, 0) for c in SUBSTANTIVE_CODES], dtype=float)
    row["llm_move_entropy"] = _entropy(counts_arr)

    early_mask, mid_mask, late_mask = _third_mask(pos)
    for mask, suffix in [(early_mask, "early"), (mid_mask, "mid"), (late_mask, "late")]:
        sub = subst_grp[mask]
        n_sub = len(sub)
        if n_sub == 0:
            row[f"llm_he_{suffix}"] = 0.0
        else:
            n_prompt = sub["code"].str.startswith("PROMPTING_").sum()
            row[f"llm_he_{suffix}"] = float(n_prompt / n_sub)
    row["llm_scaffold_slope"] = row["llm_he_late"] - row["llm_he_early"]

    n_fallback = _count_fallbacks(subst_grp["code"].tolist())
    row["llm_fallback_count"] = float(n_fallback)
    row["llm_fallback_rate"] = _safe_frac(n_fallback, max(prompting_total, 1))

    late_sub = subst_grp[late_mask]
    row["llm_giving_answer_late"] = _safe_frac((late_sub["code"] == "GIVING_ANSWER").sum(), max(len(late_sub), 1))
    return row


# Compute one session's student code rates and llm_stu_confusion_resolve
# (SELF_CORRECTING count / CONFUSION_EXPRESSED count).
#
# Args:
#     grp: student annotation records of one session (utterance_id, code).
#
# Returns:
#     Dict with llm_stu_<code> per student code and llm_stu_confusion_resolve.
def _student_features(grp: pd.DataFrame) -> dict:
    grp = grp.sort_values("utterance_id")
    total = len(grp)
    code_counts = grp["code"].value_counts()
    row: dict = {}
    for code in STUDENT_CODES:
        row[f"llm_stu_{code.lower()}"] = _safe_frac(code_counts.get(code, 0), total)

    confusion = code_counts.get("CONFUSION_EXPRESSED", 0)
    self_corr = code_counts.get("SELF_CORRECTING", 0)
    row["llm_stu_confusion_resolve"] = _safe_frac(self_corr, confusion)
    return row


def _sess_empty_row() -> dict:
    """All-zero row for a session missing tutor_whole/student annotations."""
    row = {c: 0.0 for c in LLM_SESS_COLS}
    return row


def compile_session_features(annotations: list[dict]) -> pd.DataFrame:
    """Compile the session-level LLM features from the tutor_whole and student annotations.

    Args:
        annotations: records from :func:`run_vllm` for tutor_whole and student combined.

    Returns:
        DataFrame indexed by session_id with the LLM_SESS_COLS columns; a role without
        annotations in a session gets 0.0.
    """
    if not annotations:
        return pd.DataFrame(columns=["session_id"] + LLM_SESS_COLS).set_index("session_id")
    ann = pd.DataFrame(annotations)
    tutor = ann[ann["annotation_type"] == "tutor_whole"]
    student = ann[ann["annotation_type"] == "student"]
    tutor_groups = {sid: grp for sid, grp in tutor.groupby("session_id")}
    student_groups = {sid: grp for sid, grp in student.groupby("session_id")}
    all_sessions = set(tutor_groups) | set(student_groups)

    rows = []
    for sid in all_sessions:
        row = {"session_id": sid}
        t_grp = tutor_groups.get(sid)
        if t_grp is not None and not t_grp.empty:
            row.update(_tutor_whole_features(t_grp))
        else:
            for col in LLM_SANDPIPER_COLS + LLM_TEMPORAL_COLS:
                row[col] = 0.0
        s_grp = student_groups.get(sid)
        if s_grp is not None and not s_grp.empty:
            row.update(_student_features(s_grp))
        else:
            for col in [f"llm_stu_{c.lower()}" for c in STUDENT_CODES] + ["llm_stu_confusion_resolve"]:
                row[col] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows).set_index("session_id")
    return df[LLM_SESS_COLS].fillna(0.0)


# Compile the rubric records into a frame.
#
# Args:
#     rubric_records: records from run_vllm_rubric.
#
# Returns:
#     DataFrame indexed by session_id with the LLM_RUBRIC_COLS columns (0/1).
def compile_rubric_features(rubric_records: list[dict]) -> pd.DataFrame:
    if not rubric_records:
        return pd.DataFrame(columns=["session_id"] + LLM_RUBRIC_COLS).set_index("session_id")
    df = pd.DataFrame(rubric_records).rename(columns={k: f"llm_rubric_{k}" for k in RUBRIC_KEYS})
    return df[["session_id"] + LLM_RUBRIC_COLS].fillna(0).set_index("session_id")


def compile_arc_features(arc_records: list[dict], encoder, pca) -> pd.DataFrame:
    """Compile the per-(session, objective) arc and misconception features.

    Args:
        arc_records: records from :func:`run_vllm_arc_misconception`.
        encoder: SentenceTransformer (all-MiniLM-L6-v2) used to embed the misconception
            descriptions, or None (embedding columns are then NaN).
        pca: a frozen, already-fitted sklearn PCA, applied with ``transform`` only and never
            refitted at inference (test samples must be processed independently), or None
            (embedding columns are then NaN).

    Returns:
        DataFrame indexed by (session_id, lo_id) with the LLM_ARC_COLS columns. The
        misconception columns are NaN where no misconception is present.
    """
    if not arc_records:
        return pd.DataFrame(columns=["session_id", "lo_id"] + LLM_ARC_COLS).set_index(["session_id", "lo_id"])

    rows = []
    for r in arc_records:
        confusion_count = float(r.get("confusion_count", 0) or 0)
        resolution_count = float(r.get("resolution_count", 0) or 0)
        misconception_present = bool(r.get("misconception_present", False))
        misconception_resolved = r.get("misconception_resolved")
        resolution_type = str(r.get("resolution_type", "none"))
        is_resolved = float(misconception_resolved) if (
            misconception_present and misconception_resolved is not None
        ) else float("nan")
        is_self = float(resolution_type == "student_self") if misconception_present else float("nan")
        is_tutor_corr = float(resolution_type == "tutor_corrected") if misconception_present else float("nan")
        rows.append({
            "session_id": r["session_id"], "lo_id": r["lo_id"],
            "_description": str(r.get("misconception_description", "") or ""),
            "_misconception_present": misconception_present,
            "llm_arc_confusion_count": confusion_count,
            "llm_arc_resolution_rate": _safe_frac(resolution_count, confusion_count),
            "llm_arc_final_resolved": float(str(r.get("final_state", "")) == "resolved"),
            "llm_arc_student_led": float(str(r.get("resolution_agent", "")) == "student_led"),
            "llm_misconception_present": float(misconception_present),
            "llm_misconception_resolved": is_resolved,
            "llm_misconception_student_self": is_self,
            "llm_misconception_tutor_corrected": is_tutor_corr,
        })
    out = pd.DataFrame(rows)

    embed_cols = ["llm_misconception_embed_0", "llm_misconception_embed_1",
                  "llm_misconception_embed_2", "llm_misconception_embed_3"]
    if encoder is not None and pca is not None:
        has_misconception = out["_misconception_present"].to_numpy()
        descriptions = out["_description"].tolist()
        embs = encoder.encode(descriptions, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
        embed_matrix = pca.transform(embs)  # transform only; never .fit() at inference
        for i, col in enumerate(embed_cols):
            out[col] = np.where(has_misconception, embed_matrix[:, i], float("nan"))
    else:
        for col in embed_cols:
            out[col] = float("nan")

    return out.set_index(["session_id", "lo_id"])[LLM_ARC_COLS]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
ALL_GOPT_GROUPS = frozenset({"tutor_whole", "student", "rubric", "arc_misconception"})


def annotate_and_build_gopt_llm_matrix(
    feats: pd.DataFrame,
    store,
    qwen_model_dir: str,
    encoder_dir: str,
    misconception_pca_path: "Path | str | None" = None,
    batch_size: int = 40,
    gpu_util: float = 0.5,
    tensor_parallel_size: int = 1,
    groups: frozenset[str] = ALL_GOPT_GROUPS,
) -> pd.DataFrame:
    """Annotate test transcripts with Qwen and compile the ``model_g_opt`` Sandpiper features.

    Covers the tutor_whole, student, rubric and arc/misconception groups.

    ``groups`` lets a caller disable phases for step-by-step throughput and correctness
    testing, e.g. run only {"tutor_whole", "student"} first, confirm it works and time it,
    then add "rubric", then "arc_misconception", each a separate `llm.generate()` call.
    Disabled phases skip generation (no GPU work) and their columns come back NaN or 0.0;
    every compile_* function already handles an empty record list this way, so the output
    columns do not depend on which groups ran. `main.py` always uses the default (all
    groups); the parameter only matters for development testing.

    ``batch_size=40`` matches the batch size used to annotate the training set.
    Utterance-level classification of tutor_whole/student is by far the most
    expensive part of this pipeline, so this default matters: larger batches reduce
    prompt-scheduling overhead but not the total number of generated tokens, which dominates
    wall time.

    ``gpu_util=0.5`` is conservative: on the competition GPU (1x A100 80 GB per the runtime
    docs) it leaves ~40 GB of headroom for a model that needs ~14 GB of weights, trading some
    throughput for a safety margin against running out of GPU memory while vLLM profiles and
    reserves it at initialisation.

    Args:
        feats: rows to score, with response_id, session_id, learning_objective_id and
            learning_objective.
        store: a :class:`magnificat.data.TranscriptStore` (directory-backed at inference,
            data/test_transcripts).
        qwen_model_dir: local directory of Qwen2.5-7B-Instruct (model and tokenizer).
        encoder_dir: local path to a bundled copy of sentence-transformers/all-MiniLM-L6-v2
            (e.g. ``assets/all-MiniLM-L6-v2``), never the bare model name: with a name,
            ``SentenceTransformer(...)`` tries to reach huggingface.co, which the competition
            runtime cannot (the run stops with a `NameResolutionError`).
        misconception_pca_path: pickle of the frozen PCA for the misconception embeddings;
            if None or missing, the embedding columns are NaN.
        batch_size: maximum utterances per tutor_whole/student prompt.
        gpu_util: vLLM ``gpu_memory_utilization``.
        tensor_parallel_size: vLLM tensor-parallel size.
        groups: phases to run, a subset of ``ALL_GOPT_GROUPS`` (default: all four).

    Returns:
        DataFrame indexed by response_id with LLM_SESS_COLS (36) + LLM_RUBRIC_COLS (6) +
        LLM_ARC_COLS (12) = 54 columns. Sets the environment variables
        VLLM_USE_FLASHINFER_SAMPLER, VLLM_DEEP_GEMM_WARMUP, VLLM_NO_USAGE_STATS and
        VLLM_HOST_IP as a side effect.
    """
    uniq_sessions = list(dict.fromkeys(feats["session_id"]))
    lo_pairs = list(
        feats[["session_id", "learning_objective_id", "learning_objective"]]
        .drop_duplicates(["session_id", "learning_objective_id"])
        .itertuples(index=False, name=None)
    )
    lo_pairs = [(sid, str(lo_id), str(lo_text)) for sid, lo_id, lo_text in lo_pairs]

    # ── phase 1: sentence encoder on CPU, which avoids GPU contention with vLLM
    # in this single-process live job. Loaded from a local bundled path only,
    # never by bare model name (no internet at inference).
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(encoder_dir, device="cpu")
    emb_cache: dict = {}
    df_cache: dict = {}

    # No prints anywhere in this function, including static phase markers. The
    # competition rules ask for debug prints to be removed without a content
    # qualifier, and with disqualification at stake this follows the stricter
    # reading, even for prints that carry no test content. Diagnose by local
    # reproduction, not from output in the scored run.

    # ── phase 2: vLLM (needs the tokenizer for chat templates, so build the
    # LO-window-dependent arc prompts only after the tokenizer is available) ──
    #
    # Two independent ways of avoiding FlashInfer, with the pinned
    # vllm==0.21.0+cu129:
    #
    # 1. attention_config={"backend": "TRITON_ATTN"} moves vLLM off its
    #    auto-selected FlashInfer attention backend. TRITON_ATTN rather than
    #    FLASH_ATTN because Triton is a core vLLM dependency and always
    #    present, whereas the separate `flash-attn` package is not known to be
    #    in the runtime.
    # 2. VLLM_USE_FLASHINFER_SAMPLER=0 (environment variable, set below) moves
    #    the top-k/top-p sampler off FlashInfer as well. This is a separate code
    #    path from attention (vllm/v1/sample/ops/topk_topp_sampler.py imports
    #    FlashInferBackend regardless of attention_config). Without it,
    #    FlashInfer tries to JIT-compile its sampling kernel on first use and
    #    fails with "RuntimeError: Could not find nvcc and default
    #    cuda_home='/usr/local/cuda' doesn't exist", because the runtime has no
    #    nvcc. Installing flashinfer-jit-cache (pre-built kernels) also avoids
    #    it, but that package belongs to the runtime environment, which we do
    #    not control; VLLM_USE_FLASHINFER_SAMPLER=0 is sufficient on its own.
    # Both are kept; TRITON_ATTN alone is not sufficient.
    import os as _os
    _os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Precaution, not known to be needed on the A100 platform: DeepGEMM warmup
    # only activates on Hopper/Blackwell GPUs (vllm.utils.deep_gemm.
    # is_deep_gemm_supported() checks compute capability), so it should not
    # trigger on the documented A100 target; setting it costs nothing.
    _os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    _os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    # Host IP. In the network-isolated runtime, vLLM can stall after logging
    # "[c10d] The hostname of the client socket cannot be retrieved. err=-3"
    # (EAI_AGAIN, DNS unavailable). From the vLLM source: even a single-GPU
    # (world_size=1, "uni" executor) configuration calls get_ip()
    # (vllm/utils/network_utils.py) at vllm/v1/executor/uniproc_executor.py:70
    # to build the distributed init address, and get_ip() falls back to an
    # outbound UDP "connect" to 8.8.8.8 or a DNS-dependent hostname lookup when
    # no override is set. Setting VLLM_HOST_IP to loopback skips the probing;
    # loopback is always available on a single-node job. This setting follows
    # from reading the source.
    _os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    # vLLM's log level is left at its default (INFO): a WARNING level hides the
    # progress markers (model-shard loading, the "Using FlashInfer" line,
    # generation start) that show where time is going.

    from vllm import LLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_dir)
    llm = LLM(
        model=qwen_model_dir, dtype="bfloat16", gpu_memory_utilization=gpu_util,
        max_model_len=8192, tensor_parallel_size=tensor_parallel_size, enforce_eager=True,
        attention_config={"backend": "TRITON_ATTN"},
    )

    # tutor_whole + student (session-wide) -- each independently gated by `groups`
    tutor_records: list[dict] = []
    if "tutor_whole" in groups:
        tutor_meta, tutor_prompts = build_prompts_whole(
            uniq_sessions, store, batch_size, tokenizer, "tutor", TUTOR_SYSTEM_PROMPT)
        tutor_records = run_vllm(tutor_meta, tutor_prompts, llm,
                                  _make_sampling_params(TUTOR_SCHEMA), TUTOR_SET, label="tutor_whole")
    student_records: list[dict] = []
    if "student" in groups:
        student_meta, student_prompts = build_prompts_whole(
            uniq_sessions, store, batch_size, tokenizer, "student", STUDENT_SYSTEM_PROMPT)
        student_records = run_vllm(student_meta, student_prompts, llm,
                                    _make_sampling_params(STUDENT_SCHEMA), STUDENT_SET, label="student")
    sess_feats = compile_session_features(tutor_records + student_records)

    # tutor_rubric (per-session)
    rubric_records: list[dict] = []
    if "rubric" in groups:
        rubric_meta, rubric_prompts = build_prompts_rubric(uniq_sessions, store, tokenizer)
        rubric_records = run_vllm_rubric(
            rubric_meta, rubric_prompts, llm, _make_sampling_params(RUBRIC_SCHEMA, max_tokens=128))
    rubric_feats = compile_rubric_features(rubric_records)

    # tutor_lo_arc_misconception (per session-LO pair): needs the encoder for
    # window selection, built now that the tokenizer exists
    arc_records: list[dict] = []
    if "arc_misconception" in groups:
        arc_meta, arc_prompts = build_prompts_arc_misconception(
            lo_pairs, store, tokenizer, encoder, emb_cache, df_cache=df_cache)
        arc_records = run_vllm_arc_misconception(
            arc_meta, arc_prompts, llm, _make_sampling_params(ARC_MISCONCEPTION_SCHEMA, max_tokens=192))

    pca = None
    if misconception_pca_path is not None and Path(misconception_pca_path).exists():
        import pickle
        with open(misconception_pca_path, "rb") as f:
            pca = pickle.load(f)  # frozen: .transform() only, see compile_arc_features
    arc_feats = compile_arc_features(arc_records, encoder if pca is not None else None, pca)

    # ── broadcast session/session-LO features onto response_id rows ─────────
    resp_sess = feats.set_index("response_id")[["session_id"]]
    resp_lo = feats.set_index("response_id")[["session_id", "learning_objective_id"]].copy()
    resp_lo["lo_id"] = resp_lo["learning_objective_id"].astype(str)

    sess_resp = resp_sess.join(sess_feats, on="session_id")[LLM_SESS_COLS].fillna(0.0)
    rub_resp = resp_sess.join(rubric_feats, on="session_id")[LLM_RUBRIC_COLS].fillna(0.0)
    arc_resp = (
        resp_lo.merge(arc_feats.reset_index(), on=["session_id", "lo_id"], how="left")
        .set_index(resp_lo.index)[LLM_ARC_COLS]
    )

    out = sess_resp.join(rub_resp).join(arc_resp)
    # No elapsed-time print: wall-clock duration scales with the test-set size,
    # so it is an indirect test aggregate too; main.py prints only its single
    # static "inference complete" message.
    return out
