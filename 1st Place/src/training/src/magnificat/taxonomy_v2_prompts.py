"""Prompt construction for the tutoring-move taxonomy v2.

Built from the specification in src/training/docs/39_moves_annotations_v2.md (section
numbers below refer to it). Its central rule (§3) is that no definition is ever paraphrased:
a paraphrase of NTO's codebook drops the clauses that decide a label ("specific", the
PROMPTING open-response rule). Every upstream definition is therefore read from its source
when the prompt is built:

    NTO's 15          <- src/training/docs/sandpiper_codebooks.md   (table + examples)
    doc29's tutor 7   <- magnificat.doc29_prompts.DOC29_TUTOR_PROMPT
    doc29's student 4 <- magnificat.doc29_prompts.DOC29_STUDENT_PROMPT
    sandpiper's 3     <- magnificat.sandpiper_live_infer

Only the five new tutor codes and the ten role-prefixed residuals are worded by us, because
they have no upstream source. :func:`verify_nto_verbatim` asserts that every definition read
from a source appears unchanged in the built prompt; call it before every run.

Two deviations from the specification:

1. §11 step 2 is applied. The specification gives the §5.1 precedence rule ("a question mark
   never decides the label") but its §12 prompt text does not include it. It is in
   ``_Q_RULE`` below, and ``build_tutor_prompt(q_rule=False)`` gives the specification's
   prompt text verbatim.
2. The key field is ``turn``, not ``utterance_id``. The §12 prompts were written against raw
   role-filtered utterances. Our move pipeline (``turn_text.pkl.gz``,
   ``distil_nto_moves.py``, ``nto_features.py``) is keyed by the paired turn index that
   ``build_session_dialogue``/``process_dialogue`` produce (2,368,779 rows, 22,816 sessions).
   Only the output-format line changes; no definition is touched.

No other wording is ours. The residual codes follow the decision order of §5
(:func:`_decision_order`).
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CODEBOOK = _ROOT / "docs" / "sandpiper_codebooks.md"

# --------------------------------------------------------------------------------------
# Source readers. Each returns {CODE: definition} with the source text unchanged.
# --------------------------------------------------------------------------------------


def _unwrap(s: str) -> str:
    """Collapse a backslash-continued, indented prompt block to one line.

    The source definitions are written as Python line continuations, so they carry the
    indentation of their block. Whitespace is normalised; no word is altered. Definitions are
    read, never re-typed, so no clause (such as a `NOT` clause) is lost.

    Args:
        s: the definition text as written in the source prompt.

    Returns:
        The text with every whitespace run collapsed to one space, stripped.
    """
    return re.sub(r"\s+", " ", s).strip()


def read_nto_codebook() -> tuple[dict[str, str], dict[str, str]]:
    """Read NTO's 15 tutor-move definitions and examples verbatim from the codebook file.

    Parses only the `## 1. Tutor Moves` section of src/training/docs/sandpiper_codebooks.md;
    the file also holds the TalkMoves and session-rubric codebooks, which are different
    instruments.

    Returns:
        Tuple ``(definitions, examples)``, each a dict ``{CODE: text}``.

    Raises:
        RuntimeError: if the section does not yield exactly 15 codes, or a code has no example.
    """
    text = _CODEBOOK.read_text(encoding="utf-8")
    sec = text.split("## 1. Tutor Moves")[1].split("## 2. Talk Moves")[0]

    defs: dict[str, str] = {}
    for line in sec.split("### Examples")[0].splitlines():
        m = re.match(r"^\|\s*`([A-Z_]+)`\s*\|\s*(.+?)\s*\|\s*$", line)
        if m:
            defs[m.group(1)] = m.group(2).strip()

    egs: dict[str, str] = {}
    for line in sec.split("### Examples")[1].split("### Notes")[0].splitlines():
        m = re.match(r"^-\s*`([A-Z_]+)`:\s*(.+?)\s*$", line)
        if m:
            egs[m.group(1)] = m.group(2).strip()

    if len(defs) != 15:
        raise RuntimeError(f"expected 15 NTO codes in {_CODEBOOK}, parsed {len(defs)}")
    missing = set(defs) - set(egs)
    if missing:
        raise RuntimeError(f"NTO codes without an example: {sorted(missing)}")
    return defs, egs


def _read_bullet_defs(prompt: str, codes: list[str]) -> dict[str, str]:
    """Extract `- CODE: definition` blocks from an existing prompt string.

    A block runs to the next `- CODE:` bullet, section heading, or the prompt's trailing
    "Output format" line, so unindented continuation lines that qualify a code (doc29's
    "Judge visible reasoning only...") stay attached while the host prompt's scaffolding does
    not. Without the `Output format` terminator the last bullet of a source prompt (e.g.
    `FRUSTRATION`, the last bullet of DOC29_STUDENT_PROMPT) would absorb that line without
    any error.

    Args:
        prompt: the source prompt text.
        codes: the codes to extract.

    Returns:
        Dict ``{CODE: definition}`` with whitespace normalised by :func:`_unwrap`.

    Raises:
        RuntimeError: if a code is not found in the prompt.
    """
    out: dict[str, str] = {}
    for code in codes:
        m = re.search(rf"^-\s*{code}:\s*(.*?)(?=^-\s*[A-Z_]+:|^##|^Output format|\Z)",
                      prompt, re.S | re.M)
        if not m:
            raise RuntimeError(f"could not read {code} from its source prompt")
        out[code] = _unwrap(m.group(1))
    return out


#: doc29's tutor moves that v2 keeps (its 8th code, `NONE`, is replaced by the §5 residuals)
DOC29_TUTOR_KEEP = ["NORMALIZING_DIFFICULTY", "RELEASE_HANDOFF", "TRANSFER_PROBE",
                    "SIMPLIFYING_TO_SUBPROBLEM", "MOTIVATING_RELEVANCE",
                    "SUMMARIZING_PROGRESS", "OFFERING_CHOICE"]
#: doc29 student codes v2 keeps, from the reasoning and affect fields
DOC29_STUDENT_KEEP = ["ANSWER_ONLY", "REASONING_SHOWN", "INSIGHT_DELIGHT", "FRUSTRATION"]
#: sandpiper student codes v2 retains (its others are replaced by the §5 residuals)
SANDPIPER_STUDENT_KEEP = ["CONFUSION_EXPRESSED", "SEEKING_CLARIFICATION", "SELF_CORRECTING"]


def read_doc29_tutor() -> dict[str, str]:
    """Read the seven kept interview-codebook tutor definitions from DOC29_TUTOR_PROMPT.

    Returns:
        Dict ``{CODE: definition}`` for :data:`DOC29_TUTOR_KEEP`.
    """
    from .doc29_prompts import DOC29_TUTOR_PROMPT
    return _read_bullet_defs(DOC29_TUTOR_PROMPT, DOC29_TUTOR_KEEP)


def read_doc29_student() -> dict[str, str]:
    """Read the four kept interview-codebook student definitions from DOC29_STUDENT_PROMPT.

    Returns:
        Dict ``{CODE: definition}`` for :data:`DOC29_STUDENT_KEEP`.
    """
    from .doc29_prompts import DOC29_STUDENT_PROMPT
    return _read_bullet_defs(DOC29_STUDENT_PROMPT, DOC29_STUDENT_KEEP)


def read_sandpiper_student() -> dict[str, str]:
    """Read the three kept student definitions from the Sandpiper-style student prompt.

    Returns:
        Dict ``{CODE: definition}`` for :data:`SANDPIPER_STUDENT_KEEP`.
    """
    from .sandpiper_live_infer import STUDENT_SYSTEM_PROMPT as SANDPIPER_STUDENT_PROMPT
    return _read_bullet_defs(SANDPIPER_STUDENT_PROMPT, SANDPIPER_STUDENT_KEEP)


# --------------------------------------------------------------------------------------
# Worded by us: the five new codes (§7) and the residual ladder (§5).
# --------------------------------------------------------------------------------------

NEW_TUTOR_DEFS = {
    "ASKING_QUESTION": (
        "Asks a question expecting a brief, factual or binary answer — recall, a "
        "calculation, or a comprehension check. Use when no PROMPTING_* code applies "
        "because the tutor is NOT asking for an open explanation."),
    "GUIDING_SESSION": (
        "Directs the flow of the lesson rather than the mathematics: announcing what comes "
        "next, moving between slides or tasks, allocating time."),
    "EXPLAINING_TOOL": (
        "Instructs on the platform rather than the mathematics — where to write, how to use "
        "a feature, what a control does."),
    "GIVING_PRAISE": (
        "Praises effort, progress or the student personally (\"well done\", \"good job\"). "
        "Distinct from FEEDBACK_CORRECT, which confirms a SPECIFIC answer is right."),
    "WORKED_DEMONSTRATION": (
        "Works the CURRENT problem through aloud to model the method. Distinct from "
        "GIVING_EXAMPLE, which uses a DIFFERENT scenario or set of numbers."),
}


def _residuals(role: str) -> dict[str, str]:
    """Return the five role-prefixed residual codes, in decision order (§5).

    Args:
        role: ``"tutor"`` or ``"student"``; used upper-cased as the code prefix.

    Returns:
        Ordered dict ``{CODE: definition}``.
    """
    r = role.upper()
    return {
        f"{r}_UNINTELLIGIBLE": ("The turn is [unclear] or [inaudible]. A transcription "
                                "failure, not a move — never a reason to reach for a "
                                "pedagogical code."),
        f"{r}_TECHNICAL": "Connection or audio (\"can you hear me?\", \"you're frozen\").",
        f"{r}_SOCIAL": "Greetings, closings, rapport, small talk.",
        f"{r}_OTHER": ("The turn IS pedagogically meaningful, but no code above fits. "
                       "Choosing this reports that the taxonomy is incomplete — use it "
                       "rather than forcing a poor fit."),
        f"{r}_NO_MOVE": ("Nothing pedagogically notable. The FALLBACK: only after rejecting "
                         "everything above."),
    }


def _decision_order(role: str) -> str:
    """Return the "How to choose" prompt section for one role (§5).

    Args:
        role: ``"tutor"`` or ``"student"``.

    Returns:
        The prompt section as a string.
    """
    r = role.upper()
    return f"""# How to choose (in this order — do not skip ahead)
1. An instructional move from the list, if one genuinely fits.
2. {r}_UNINTELLIGIBLE, if the turn is [unclear]/[inaudible].
3. {r}_TECHNICAL, for connection or audio.
4. {r}_SOCIAL, for greetings, rapport or small talk.
5. {r}_OTHER, if the turn IS pedagogically meaningful but nothing above fits.
6. {r}_NO_MOVE — the fallback, and only once 1-5 have all been rejected.

Steps 5 and 6 are different. {r}_OTHER means "this matters and the list is missing it";
{r}_NO_MOVE means "nothing here". Do not use {r}_NO_MOVE to avoid a hard decision."""


# §5.1, applied per §11 step 2: a precedence rule that keeps ASKING_QUESTION from becoming a
# catch-all. The outer bullets are NTO's own instructions, verbatim (§2).
_Q_RULE = """
- A QUESTION MARK NEVER DECIDES THE LABEL. Classify by what the turn DOES, not by its
  grammatical form. Many questions are really something else: "can you now solve question b?"
  transfers the work, so it is RELEASE_HANDOFF; "shall we move on to the next slide?" directs
  the lesson, so it is GUIDING_SESSION. ASKING_QUESTION is the LAST resort among
  interrogative-looking codes, not the first — use it only when the function is eliciting a
  brief factual answer and no other code fits. "What's 7 times 8?" is ASKING_QUESTION."""

# `q_rule=False` reproduces the §12 clarifications block verbatim, i.e. the prompt as the
# specification gave it, without the §5.1 rule. The labels behind the shipped model
# use `q_rule=True`.
_TUTOR_CLARIFICATIONS_HEAD = """# Clarifications
- PROMPTING is when a tutor expects an open response answer like an explanation, NOT when the
  tutor expects a brief or binary answer (e.g., yes or no). A closed question is
  ASKING_QUESTION, not a PROMPTING_* code."""
_TUTOR_CLARIFICATIONS_TAIL = """
- Code the tutor only. Do not code student utterances."""

# Line breaks here are those of §12, not ours. Do not rewrap these lines: the built prompt
# must keep matching the document's fenced block.
_STUDENT_CLARIFICATIONS_HEAD = """# Key distinctions
- ANSWER_ONLY vs REASONING_SHOWN is about VISIBLE thinking, not correctness. A correct bare answer is ANSWER_ONLY; a wrong answer with working is REASONING_SHOWN."""
_STUDENT_CLARIFICATIONS_TAIL = """
- A backchannel ("yeah", "mm-hmm") is STUDENT_NO_MOVE, not an answer.
- Reading the problem aloud without comment is STUDENT_NO_MOVE."""

# §11 step 2: "Add the §5.1 precedence rule to both prompts." The student form is not a
# mechanical copy of the tutor form: the student pool has its own form-over-function trap,
# and a sharper one. SEEKING_CLARIFICATION is defined by
# grammatical form ("student asks a question about the material or task"), while
# `stance: hedged` explicitly covers an answer "delivered as a question". The two overlap
# exactly on "is it 12?", which is an answer, not a request for information. The wording is
# ours (§5.1 states the principle but gives only tutor-side examples), so it is flagged like
# NEW_TUTOR_DEFS.
_Q_RULE_STUDENT = """
- A QUESTION MARK NEVER DECIDES THE LABEL. Classify by what the turn DOES, not by its
  grammatical form. "Is it 12?" is an ANSWER offered with uncertainty — ANSWER_ONLY with
  stance "hedged" — NOT SEEKING_CLARIFICATION. Use SEEKING_CLARIFICATION only for a genuine
  request for information about the material or task ("what do you mean by numerator?")."""

_SECOND_LABEL_TUTOR = """# Second label
Nearly always emit one code. Add "code_2" ONLY when the turn genuinely does two things (e.g. praising and moving on). "code" is the move that most shapes what the student does next."""

# The §12 student block has no "code is the move that most shapes..." sentence, and it is not
# added here: the student pool has no "shapes what the student does next" semantics.
_SECOND_LABEL_STUDENT = """# Second label
Nearly always emit one code. Add "code_2" ONLY when the turn genuinely does two things."""


# --------------------------------------------------------------------------------------
# Pool assembly
# --------------------------------------------------------------------------------------


def tutor_pool(extend: bool = True) -> dict[str, str]:
    """Assemble the v2 tutor pool, in decision-relevant order.

    ``extend=False`` gives the 20-code fidelity pool of §10 (NTO's 15 plus the residual
    ladder), without doc29's 7 and the five new codes. The labels behind the shipped
    model use ``extend=True``.

    Args:
        extend: include the doc29 tutor codes and the five new codes (the full v2 pool).

    Returns:
        Ordered dict ``{CODE: definition}``: NTO codes, then (if ``extend``) doc29 and new
        codes, then the tutor residuals.
    """
    nto, _ = read_nto_codebook()
    pool = dict(nto)
    if extend:
        pool.update(read_doc29_tutor())
        pool.update(NEW_TUTOR_DEFS)
    pool.update(_residuals("tutor"))
    return pool


def student_pool() -> dict[str, str]:
    """Assemble the v2 student pool: doc29 codes, Sandpiper student codes, then residuals.

    Returns:
        Ordered dict ``{CODE: definition}``.
    """
    pool = dict(read_doc29_student())
    pool.update(read_sandpiper_student())
    pool.update(_residuals("student"))
    return pool


def _format_defs(pool: dict[str, str], egs: dict[str, str] | None = None) -> str:
    """Format a code pool as the prompt's "Definitions" bullet list.

    Args:
        pool: ordered ``{CODE: definition}``.
        egs: optional ``{CODE: example}``; an example is added under its code's bullet.

    Returns:
        One ``- CODE: definition`` line per code (plus an indented ``e.g.`` line if any).
    """
    egs = egs or {}
    lines = []
    for code, d in pool.items():
        lines.append(f"- {code}: {d}")
        if code in egs:
            lines.append(f"    e.g. {egs[code]}")
    return "\n".join(lines)


def build_tutor_prompt(extend: bool = True, q_rule: bool = True,
                       null_secondary: bool = True) -> str:
    """Build the full tutor system prompt (§12) from the source definitions.

    Args:
        extend: use the full v2 pool; False gives the 20-code fidelity pool (see
            :func:`tutor_pool`).
        q_rule: include the §5.1 precedence rule. False reproduces the specification's
            prompt exactly.
        null_secondary: write ``null`` (True) or ``NONE`` (False) as the empty ``code_2``
            value in the output-format line.

    Returns:
        The prompt text.
    """
    sec = "null" if null_secondary else "NONE"
    pool = tutor_pool(extend=extend)
    _, egs = read_nto_codebook()
    return f"""You are an expert educational discourse analyst. Classify each TUTOR turn in a 1:1 maths tutoring session. Return ONLY a JSON array — no outer object, no explanation.

These are ASR transcripts of spoken sessions, so expect disfluency, [unclear] markers and short backchannels.

# Allowed codes
{", ".join(pool)}

# Definitions
{_format_defs(pool, egs)}

{_decision_order("tutor")}

{_TUTOR_CLARIFICATIONS_HEAD}{_Q_RULE if q_rule else ""}{_TUTOR_CLARIFICATIONS_TAIL}

{_SECOND_LABEL_TUTOR}

# performed_by
For every instructional code, also say whether the tutor DOES the move or asks the STUDENT to do it: "tutor_does" or "student_elicited". Use "tutor_does" for residual codes.

Output format: [{{"turn": <int>, "code": "<CODE>", "code_2": "<CODE or {sec}>", "performed_by": "<tutor_does|student_elicited>"}}, ...]"""


def build_student_prompt(q_rule: bool = True, null_secondary: bool = True) -> str:
    """Build the full student system prompt (§12) from the source definitions.

    Args:
        q_rule: include the student form of the §5.1 precedence rule; False omits it
            (see :func:`build_tutor_prompt`).
        null_secondary: write ``null`` (True) or ``NONE`` (False) as the empty ``code_2``
            value in the output-format line.

    Returns:
        The prompt text.
    """
    sec = "null" if null_secondary else "NONE"
    pool = student_pool()
    return f"""You are an expert educational discourse analyst. Classify each STUDENT turn in a 1:1 maths tutoring session. Return ONLY a JSON array — no outer object, no explanation.

These are ASR transcripts of spoken sessions. Judge what the student is doing, not how polished it sounds.

# Allowed codes
{", ".join(pool)}

# Definitions
{_format_defs(pool)}

{_decision_order("student")}

{_STUDENT_CLARIFICATIONS_HEAD}{_Q_RULE_STUDENT if q_rule else ""}{_STUDENT_CLARIFICATIONS_TAIL}

# stance
For every turn that asserts something, also give the student's stance on their OWN claim: "hedged" (uncertainty markers, or delivered as a question), "assertive" (stated flatly as fact), or "unmarked" (neither — the DEFAULT and most common). Use "na" when the turn asserts nothing. Stance is independent of correctness.

{_SECOND_LABEL_STUDENT}

Output format: [{{"turn": <int>, "code": "<CODE>", "code_2": "<CODE or {sec}>", "stance": "<hedged|unmarked|assertive|na>"}}, ...]"""


# --------------------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------------------

STANCE_VALUES = ["hedged", "unmarked", "assertive", "na"]
PERFORMED_BY_VALUES = ["tutor_does", "student_elicited"]


def verify_nto_verbatim(prompt: str | None = None, extend: bool = True) -> int:
    """Check that every definition read from a source appears unchanged in the tutor prompt.

    Per §3, this check backs the claim that the prompt uses NTO's validated codebook verbatim.
    Call it before every run.

    Args:
        prompt: the prompt to check; built with :func:`build_tutor_prompt` if None.
        extend: whether the prompt is the full pool (also checks the doc29 definitions) or
            the 20-code fidelity pool.

    Returns:
        The number of definitions and examples checked.

    Raises:
        AssertionError: if a definition or example is missing or altered.
    """
    prompt = build_tutor_prompt(extend=extend) if prompt is None else prompt
    nto, egs = read_nto_codebook()
    checked = 0
    for code, d in nto.items():
        if d not in prompt:
            raise AssertionError(f"NTO definition altered for {code}:\n  {d}")
        if egs[code] not in prompt:
            raise AssertionError(f"NTO example altered for {code}:\n  {egs[code]}")
        checked += 2
    if extend:  # the fidelity pool (20 codes) carries no doc29 definitions to check
        for code, d in read_doc29_tutor().items():
            if d not in prompt:
                raise AssertionError(f"doc29 definition altered for {code}:\n  {d}")
            checked += 1
    return checked


def verify_student_verbatim(prompt: str | None = None) -> int:
    """Check that every source-read student definition appears unchanged in the prompt.

    Args:
        prompt: the prompt to check; built with :func:`build_student_prompt` if None.

    Returns:
        The number of definitions checked.

    Raises:
        AssertionError: if a definition is missing or altered.
    """
    prompt = build_student_prompt() if prompt is None else prompt
    checked = 0
    for src in (read_doc29_student(), read_sandpiper_student()):
        for code, d in src.items():
            if d not in prompt:
                raise AssertionError(f"student definition altered for {code}:\n  {d}")
            checked += 1
    return checked


def schema(codes: list[str], extra: dict) -> dict:
    """Build the guided-JSON schema for one annotation pass.

    `code_2` is nullable: it is usually absent, so the schema must allow its absence without
    also letting `code` go missing.

    Args:
        codes: the allowed codes, used as the enum of ``code`` and ``code_2``.
        extra: additional property schemas (``performed_by`` or ``stance``).

    Returns:
        A JSON-schema dict for an array of per-turn label objects.
    """
    props = {"turn": {"type": "integer"},
             "code": {"type": "string", "enum": codes},
             "code_2": {"type": ["string", "null"], "enum": codes + [None]}}
    props.update(extra)
    return {"type": "array",
            "items": {"type": "object", "properties": props,
                      "required": ["turn", "code"], "additionalProperties": False}}


def tutor_schema(extend: bool = True) -> dict:
    """Return the guided-JSON schema for the tutor pass (see :func:`schema`)."""
    return schema(list(tutor_pool(extend=extend)),
                  {"performed_by": {"type": "string", "enum": PERFORMED_BY_VALUES}})


def student_schema() -> dict:
    """Return the guided-JSON schema for the student pass (see :func:`schema`)."""
    return schema(list(student_pool()),
                  {"stance": {"type": "string", "enum": STANCE_VALUES}})


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "tutor"
    if which == "student":
        print(build_student_prompt())
        print(f"\n--- verified {verify_student_verbatim()} student definitions ---",
              file=sys.stderr)
    else:
        ext = which != "fidelity"
        print(build_tutor_prompt(extend=ext))
        print(f"\n--- verified {verify_nto_verbatim(extend=ext)} tutor definitions, "
              f"{len(tutor_pool(extend=ext))} codes ---", file=sys.stderr)
