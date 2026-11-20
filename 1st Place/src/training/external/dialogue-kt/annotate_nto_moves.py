"""NTO tutor-move annotation with vLLM and guided JSON.

Labels every tutor turn of every Trace the Ace training session with the NTO Tutor Move
Taxonomy (Table 1 of arXiv:2603.05778, transcribed verbatim in
src/training/scripts/sandpiper/nto_codebook.py). A turn can get several codes. Guided JSON
(vLLM's StructuredOutputsParams) enforces the output schema, so parse failures are close to
zero; with SO_OFF=1 (see anno_json.py) the reply is parsed
tolerantly and the code vocabulary is audited instead.

Annotator: Qwen/Qwen3-32B, greedy, thinking disabled. The NTO codes it writes have no human
validation.

Sharding and resume: ANNO_SHARD="i/N" -> results/nto_moves_shard<i>.csv (session_id, turn,
codes '|'-joined, '' = no code). Sessions already in the file are skipped on restart.
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
from nto_codebook import ALL_CODES, codebook_text  # noqa: E402
import anno_json as _aj  # noqa: E402

MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
TP = int(os.environ.get("ANNO_TP", "1"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "24"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "2000"))
SHARD_I, SHARD_N = (int(x) for x in os.environ.get("ANNO_SHARD", "0/1").split("/"))
N_SESSIONS = int(os.environ.get("ANNO_N_SESSIONS", "0"))
OUT = f"results/nto_moves_shard{SHARD_I}.csv"
BLOCK = 48          # sessions per generate() call
DATA = f"{MAGNIFICAT}/data"
SEED = 221

SYSTEM = f"""You are an expert annotator of one-on-one math tutoring transcripts. Label each
TUTOR turn with tutor-move codes from the National Tutoring Observatory (NTO) Tutor Move
Taxonomy. The codebook (code: definition), verbatim:

{codebook_text()}

Rules:
- Label ONLY the tutor turns given; consider the surrounding dialogue as context.
- A turn may get MULTIPLE codes (each distinct move present), or an empty list if none apply.
- "Probing" codes are categorical/yes-no questions; "prompting" codes are open-ended questions.
- Transcripts are noisy voice transcriptions; [unclear] marks inaudible speech. Judge from
  what is present; do not invent content.
Return ONLY the JSON object required by the schema: {{"labels": [{{"turn": <int>,
"codes": [<code>, ...]}}, ...]}} with one entry per tutor turn in the chunk."""

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
        ``((sid, tutor_turn_numbers), user_prompt)`` for every window that has a tutor turn;
        ``tutor_turn_numbers`` is the set of turns the reply should label.
    """
    for s in range(0, len(turns), CHUNK):
        lo = max(0, s - 2)
        window = turns[lo:s + CHUNK]
        tutor_turns = [t["turn"] for t in turns[s:s + CHUNK] if t.get("teacher")]
        if not tutor_turns:
            continue
        lines = []
        for t in window:
            if t.get("teacher"):
                lines.append(f"Tutor Turn {t['turn']}: {t['teacher']}")
            if t.get("student"):
                lines.append(f"Student Turn {t['turn']}: {t['student']}")
        user = ("[DIALOGUE CHUNK]\n" + "\n".join(lines) +
                f"\n[END CHUNK]\n\nLabel tutor turns {tutor_turns} with NTO codes.")
        yield (sid, set(tutor_turns)), user


def session_order():
    """Order the training sessions: one per learning objective first, then the rest shuffled.

    Returns:
        A list of session ids (seed SEED), so a capped run covers every objective first.
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
    print(f"[nto shard {SHARD_I}/{SHARD_N}] sessions to annotate: {len(order)}", flush=True)

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
            # `extract_json` instead of a bare json.loads: with SO_OFF=1 the model may add
            # fences, a preamble or trailing text around its JSON. It returns None on
            # truncation, so a cut chunk is counted as a failure, not half read.
            obj = _aj.extract_json(o.outputs[0].text)
            try:
                if obj is None:
                    raise ValueError("unparseable")
                for lab in obj["labels"]:
                    t = int(lab["turn"])
                    if t in want_turns:
                        # audit.check drops unknown codes and records them; without the schema
                        # enum nothing else constrains the vocabulary.
                        rows.append({"session_id": sid, "turn": t,
                                     "codes": "|".join(audit.check(lab.get("codes", [])))})
            except Exception:
                n_fail += 1
        n_chunks += len(prompts)
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
        print(f"  sessions {min(bstart + BLOCK, len(order))}/{len(order)}  chunks {n_chunks}  "
              f"parse_fail {n_fail}", flush=True)
    rate = n_fail / max(1, n_chunks)
    print(f"DONE shard {SHARD_I}/{SHARD_N}: chunks {n_chunks}, parse_fail {n_fail} "
          f"({rate:.2%}) -> {OUT}", flush=True)
    n_bad = audit.report(f" [nto shard {SHARD_I}]")
    # A constrained-decoding failure shows up as ~100% parse_fail with a plausible CSV still on
    # disk (see the anno_json module docstring), so such a run exits non-zero.
    if rate > 0.05 or n_bad:
        print("STOP: more than 5% of chunks failed to parse, or out-of-vocabulary codes are above "
              "the tolerance; do not use this run's labels.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
