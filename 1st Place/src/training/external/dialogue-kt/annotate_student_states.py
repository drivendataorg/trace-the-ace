"""Student-state annotation with vLLM and guided JSON.

Labels student turns with an LLM annotator whose labels are then distilled into a small
classifier (distil_student_states.py, the `studst_*` block). The per-turn correctness block
(`recdst_*`, distil_turn_correctness.py) and the tutor-move block (`ntodst_*`,
distil_nto_moves.py) are built the same way; this script covers the student side.

Most codes in the codebook below name a construct that the E2 regex features also count
(self-explanation, confusion, hedging, deep vs shallow help-seeking, resolution,
self-correction, minimal engagement), and their definitions name that feature. Three codes have
no E2 twin (`SHOWS_METHOD`, `GUESSES`, `OFF_TASK`) and are marked as such.

Annotator: Qwen/Qwen3-32B, the same model that produced the NTO codes; greedy, thinking
disabled, guided JSON so parse failures are close to zero. Apache-2.0, so the distilled child
is prize-eligible.

The script annotates a sample of sessions. The distilled child is what ships and builds the
features; it needs enough labelled turns to fit a TF-IDF + LR, not the whole corpus.
`ANNO_N_SESSIONS` caps each shard, and `session_order()` puts one session per learning
objective first, so a capped run covers all 398 objectives before it repeats any. Sessions
outside the sample are scored by the fitted child at build time, the same condition as at
inference. No fold can leak `is_correct`: every target here is the annotation.

Sharding and resume: ANNO_SHARD="i/N" -> results/student_states_shard<i>.csv (session_id,
turn, codes '|'-joined, '' = no code). Sessions already in the file are skipped on restart.
Environment: ANNO_MODEL (Qwen/Qwen3-32B), ANNO_N_SESSIONS (0 = all), ANNO_CHUNK_TURNS=24,
ANNO_MAXTOK=2000, ANNO_TP=1, ANNO_DTYPE=bfloat16, plus the anno_json.engine_kwargs variables.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "magnificat", "scripts", "sandpiper"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "sandpiper"))  # repo-relative
from build_traceace_kt import build_session_dialogue, find_one, MAGNIFICAT  # noqa: E402
import anno_json as _aj  # noqa: E402

MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
TP = int(os.environ.get("ANNO_TP", "1"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "24"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "2000"))
SHARD_I, SHARD_N = (int(x) for x in os.environ.get("ANNO_SHARD", "0/1").split("/"))
N_SESSIONS = int(os.environ.get("ANNO_N_SESSIONS", "0"))
OUT = f"results/student_states_shard{SHARD_I}.csv"
BLOCK = 48          # sessions per generate() call
DATA = f"{MAGNIFICAT}/data"
SEED = 221

#: (code, definition). Most definitions name the E2 regex feature that counts the same
#: construct; three say that the code has no E2 twin.
CODEBOOK = [
    ("EXPLAINS_WHY",
     "The student gives a REASON or justification for a step or answer — why it works, "
     "what rule applies, or a causal account. Re-measures E2's `self_explanation_rate` / "
     "`student_explains_rate` (its two strongest columns)."),
    ("SHOWS_METHOD",
     "The student states the PROCEDURE or steps they carried out, without justifying why "
     "those steps are correct. No E2 twin — the E2 regexes cannot separate 'I multiplied "
     "then divided' from 'I multiplied because...'."),
    ("ANSWER_ONLY",
     "The student gives only a value, expression or choice, with no method and no reason."),
    ("CONFUSED",
     "The student signals not understanding, being stuck, or losing the thread. "
     "Re-measures `confusion_rate`."),
    ("HEDGED",
     "The student asserts something with explicit uncertainty ('maybe', 'I think', 'not "
     "sure', rising doubt). Re-measures `hedge_rate`."),
    ("INSIGHT",
     "The student signals newly reached understanding — 'oh I see', 'that makes sense "
     "now', a correction of their own mental model. Re-measures `resolution_rate` / "
     "`recovered` / `ends_resolved`."),
    ("ASKS_CONCEPTUAL",
     "The student asks WHY or HOW something works, or for the reasoning behind a step. "
     "Re-measures `help_deep_rate`."),
    ("ASKS_VERIFICATION",
     "The student asks whether their answer is right, or what to do next, without asking "
     "for reasoning. Re-measures `help_shallow_rate`."),
    ("SELF_CORRECTS",
     "The student spontaneously revises their own earlier answer or method, without the "
     "tutor having told them it was wrong. Re-measures the student half of E2's repair "
     "arc (`recovered`)."),
    ("GUESSES",
     "The student says or clearly indicates they are guessing rather than reasoning. "
     "No E2 twin."),
    ("MINIMAL",
     "A content-free reply: bare acknowledgement, filler, or a one-word yes/no carrying "
     "no mathematical content. Re-measures `ack_only_rate` / `substantive_turn_rate` "
     "(which E2 measures by word count, not by content)."),
    ("OFF_TASK",
     "Social or off-topic talk not about the mathematics or the session. No E2 twin."),
]
ALL_CODES = [c for c, _ in CODEBOOK]


def codebook_text():
    """Return the codebook as ``- CODE: definition`` lines for the system prompt."""
    return "\n".join(f"- {c}: {d}" for c, d in CODEBOOK)


SYSTEM = f"""You are an expert annotator of one-on-one math tutoring transcripts. Label each
STUDENT turn with codes describing what the student did in that turn. The codebook
(code: definition), verbatim:

{codebook_text()}

Rules:
- Label ONLY the student turns given; consider the surrounding dialogue as context.
- A turn may get MULTIPLE codes (each distinct thing the student did), or an empty list
  if none apply.
- Judge the student's turn on its own content. Do NOT label a turn EXPLAINS_WHY just
  because the tutor asked for a reason — the student must actually give one.
- ANSWER_ONLY, SHOWS_METHOD and EXPLAINS_WHY form a ladder of increasing reasoning: use
  the highest one the turn actually reaches, and do not add ANSWER_ONLY alongside them.
- Correctness is NOT what is being coded here. A wrong answer with a reason is still
  EXPLAINS_WHY.
- Transcripts are noisy voice transcriptions; [unclear] marks inaudible speech. Judge from
  what is present; do not invent content.
Return ONLY the JSON object required by the schema: {{"labels": [{{"turn": <int>,
"codes": [<code>, ...]}}, ...]}} with one entry per student turn in the chunk."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn": {"type": "integer"},
                    "codes": {"type": "array", "items": {"type": "string", "enum": ALL_CODES}},
                },
                "required": ["turn", "codes"],
            },
        }
    },
    "required": ["labels"],
}


def chunk_dialogue(sid, turns):
    """Split a session into prompts of at most CHUNK turns each.

    Each window also shows the two turns before it as context.

    Args:
        sid: session id.
        turns: the session's turn list from ``build_session_dialogue``.

    Yields:
        ``((sid, student_turn_numbers), user_prompt)`` for every window that has a student
        turn; ``student_turn_numbers`` is the set of turns the reply should label.
    """
    for s in range(0, len(turns), CHUNK):
        lo = max(0, s - 2)
        window = turns[lo:s + CHUNK]
        student_turns = [t["turn"] for t in turns[s:s + CHUNK] if t.get("student")]
        if not student_turns:
            continue
        lines = []
        for t in window:
            if t.get("teacher"):
                lines.append(f"Tutor Turn {t['turn']}: {t['teacher']}")
            if t.get("student"):
                lines.append(f"Student Turn {t['turn']}: {t['student']}")
        user = ("[DIALOGUE CHUNK]\n" + "\n".join(lines) +
                f"\n[END CHUNK]\n\nLabel student turns {student_turns} with student-state codes.")
        yield (sid, set(student_turns)), user


def session_order():
    """Order the training sessions: one per learning objective first, then the rest shuffled.

    A capped run therefore covers all 398 objectives before it repeats any. The order and
    SEED are the same as in the NTO pass, so the two label sets cover the same sessions.

    Returns:
        A list of session ids.
    """
    import random
    feats = pd.read_csv(find_one(f"{DATA}/train_features*.csv"))
    rng = random.Random(SEED)
    first = feats.groupby("learning_objective_id", group_keys=False).head(1)
    sids = list(dict.fromkeys(first["session_id"]))
    rest = [s for s in feats["session_id"].unique() if s not in set(sids)]
    rng.shuffle(rest)
    return sids + rest


def main():
    """Annotate this shard's sessions and append the codes to OUT.

    Returns:
        None. Appends ``session_id, turn, codes`` rows to OUT and exits with status 2 if more
        than 5% of chunks fail to parse or more than 0.5% of the emitted codes are out of
        vocabulary (``anno_json`` audit tolerance).
    """
    import json
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["session_id"].unique())
    order = [s for i, s in enumerate(session_order()) if i % SHARD_N == SHARD_I]
    if N_SESSIONS:
        order = order[:N_SESSIONS]
    order = [s for s in order if s not in done]
    print(f"[student shard {SHARD_I}/{SHARD_N}] sessions to annotate: {len(order)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    # Set ANNO_DTYPE=auto for an FP8 checkpoint; bfloat16 would dequantise it.
    llm = LLM(model=MODEL, tensor_parallel_size=TP,
              dtype=os.environ.get("ANNO_DTYPE", "bfloat16"),
              max_model_len=16384, gpu_memory_utilization=0.92,
              **_aj.engine_kwargs())
    sp = _aj.make_sampling_params(SCHEMA, MAXTOK)
    audit = _aj.VocabAudit(ALL_CODES)

    n_chunks = n_fail = 0
    for bstart in range(0, len(order), BLOCK):
        sids = order[bstart:bstart + BLOCK]
        prompts, metas = [], []
        for sid in sids:
            path = f"{DATA}/train_transcripts/{sid}.csv"
            if not os.path.exists(path):
                continue
            turns = build_session_dialogue(pd.read_csv(path))
            if not turns:
                continue
            for meta, user in chunk_dialogue(sid, turns):
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
                metas.append(meta)
        if not prompts:
            continue
        outs = llm.generate(prompts, sp)
        rows = []
        for (sid, want_turns), o in zip(metas, outs):
            # Not a bare json.loads (see annotate_nto_moves): with SO_OFF=1 the model may add
            # fences, a preamble or trailing text around its JSON, and a truncated chunk must be
            # counted as a failure rather than half read.
            obj = _aj.extract_json(o.outputs[0].text)
            try:
                if obj is None:
                    raise ValueError("unparseable")
                for lab in obj["labels"]:
                    t = int(lab["turn"])
                    if t in want_turns:
                        rows.append({"session_id": sid, "turn": t,
                                     "codes": "|".join(audit.check(lab.get("codes", [])))})
            except Exception:
                n_fail += 1
        n_chunks += len(prompts)
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  sessions {min(bstart + BLOCK, len(order))}/{len(order)}  chunks {n_chunks}  "
              f"parse_fail {n_fail}", flush=True)
    rate = n_fail / max(1, n_chunks)
    _n_bad = audit.report(f" [studst shard {SHARD_I}]")
    if rate > 0.05 or _n_bad:
        print("STOP: more than 5% of chunks failed to parse, or out-of-vocabulary codes are above "
              "the tolerance; do not use this run's labels.", flush=True)
        raise SystemExit(2)
    print(f"DONE shard {SHARD_I}/{SHARD_N}: chunks {n_chunks}, parse_fail {n_fail} -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
