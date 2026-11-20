"""Annotate the corpus with the taxonomy-v2 tutoring-move codebook (vLLM, guided JSON).

Labels every tutor turn and every student turn of the training sessions with the v2 pools
built by :mod:`magnificat.taxonomy_v2_prompts` (32 tutor codes, 12 student codes; a primary and
an optional secondary code, plus `performed_by` for tutor turns and `stance` for student
turns). The taxonomy and its validation are described in
`src/training/docs/39_moves_annotations_v2.md`.

Wherever possible it matches `annotate_nto_moves.py` (in external/dialogue-kt/): same session
order, same 2-turn context overlap, same greedy decoding, same `turn` keying.

Decoding is guided JSON. Under vLLM with `StructuredOutputsParams` the label vocabulary is
enforced at decode time, so:

  - off-enum codes cannot occur: the grammar has no such tokens;
  - malformed objects cannot occur: the grammar closes them;
  - `stance` and `performed_by` are schema enums, so an illegal value cannot be generated
    rather than being filtered afterwards.

`VocabAudit` still runs: it costs nothing, and SO_OFF=1 (plain greedy sampling, no grammar)
removes the guarantee. A run that trips the audit exits with status 2.

Interleaving: both pools of a block go into one `generate()` call, each request with its own
`SamplingParams`, so the two schemas share a batch (vLLM accepts `Sequence[SamplingParams]`).
Every checkpoint therefore holds both pools for every session it covers. Merging them also
doubles the batch width.

Env: ANNO_MODEL, ANNO_ARM (ext|fidelity|noq), ANNO_ROLES (tutor,student), ANNO_SHARD=i/N,
     ANNO_N_SESSIONS (0 = all), ANNO_CHUNK_TURNS, ANNO_BLOCK, ANNO_MAXTOK, ANNO_TP, ANNO_DTYPE,
     ANNO_NULL_SECONDARY (1 = §12's `null`, 0 = NONE sentinel), ANNO_OUTDIR.
Out: <outdir>/moves_v2_{arm}_{role}_shard{i}.csv  (session_id, turn, code, code_2, aux)

Usage (reproduce.py step 02e, vLLM env, from build/workspace/; ~2-2.5 h on 2x B200; the
command was reconstructed, the original launcher was not kept):
  ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 ANNO_ARM=ext ANNO_ROLES=tutor,student \
  ANNO_SHARD=0/1 ANNO_N_SESSIONS=3072 ANNO_OUTDIR=artifacts/v2labels/122b \
    python scripts/annotate_v2_moves.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "sandpiper"), os.path.join(os.path.dirname(HERE), "external", "dialogue-kt"),
          os.path.join(os.path.dirname(HERE), "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_traceace_kt import build_session_dialogue, find_one, MAGNIFICAT  # noqa: E402
import anno_json as _aj  # noqa: E402
from magnificat import taxonomy_v2_prompts as T  # noqa: E402

MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
ARM = os.environ.get("ANNO_ARM", "ext")   # "ext" (32 codes) | "fidelity" (20) | "noq" (32, no §5.1 rule)
ROLES = [r for r in os.environ.get("ANNO_ROLES", "tutor,student").split(",") if r]
TP = int(os.environ.get("ANNO_TP", "1"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "24"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "3000"))
SHARD_I, SHARD_N = (int(x) for x in os.environ.get("ANNO_SHARD", "0/1").split("/"))
N_SESSIONS = int(os.environ.get("ANNO_N_SESSIONS", "0"))
OUTDIR = os.environ.get("ANNO_OUTDIR", "results")
BLOCK = int(os.environ.get("ANNO_BLOCK", "48"))   # sessions per generate() call; larger = better vLLM batching
DATA = f"{MAGNIFICAT}/data"
SEED = 221        # identical to annotate_nto_moves.py: same session order as the NTO pass

#: The §12 prompts emit `code_2: null`. A JSON-Schema union type is the one construct the grammar
#: backends disagree about, so `ANNO_NULL_SECONDARY=0` falls back to a plain string enum with an
#: explicit NONE sentinel, dropped at parse time. The data are the same either way (both mean
#: "no secondary code"); only the wire form differs. The default is §12's `null`; the fallback is
#: for a backend that fails to compile the union.
NONE_SENTINEL = "NONE"
NULL_SECONDARY = os.environ.get("ANNO_NULL_SECONDARY", "1") == "1"


def pass_spec(role: str):
    """Build the prompt and the guided-decoding schema for one pool.

    Verifies that the codebook definitions in the prompt are verbatim before returning.

    Args:
        role: "tutor" or "student".

    Returns:
        ``(system_prompt, json_schema, allowed_codes, aux_field)``, where aux_field is
        "performed_by" (tutor) or "stance" (student).
    """
    if role == "tutor":
        extend = ARM != "fidelity"
        # ARM=noq removes the §5.1 rule.
        prompt = T.build_tutor_prompt(extend=extend, q_rule=ARM != "noq",
                                      null_secondary=NULL_SECONDARY)
        n = T.verify_nto_verbatim(prompt, extend=extend)      # §3 check, before every run
        codes, aux, vals = list(T.tutor_pool(extend=extend)), "performed_by", T.PERFORMED_BY_VALUES
    elif role == "student":
        prompt = T.build_student_prompt(q_rule=ARM != "noq",
                                        null_secondary=NULL_SECONDARY)
        n = T.verify_student_verbatim(prompt)
        codes, aux, vals = list(T.student_pool()), "stance", T.STANCE_VALUES
    else:
        raise SystemExit(f"unknown role {role!r}")
    schema = {"type": "array", "items": {
        "type": "object",
        "properties": {"turn": {"type": "integer"},
                       "code": {"type": "string", "enum": codes},
                       "code_2": ({"anyOf": [{"type": "string", "enum": codes},
                                             {"type": "null"}]} if NULL_SECONDARY else
                                  {"type": "string", "enum": codes + [NONE_SENTINEL]}),
                       aux: {"type": "string", "enum": vals}},
        "required": ["turn", "code"], "additionalProperties": False}}
    print(f"[{role}] {len(codes)} codes, {n} source definitions verified verbatim", flush=True)
    return prompt, schema, codes, aux


def chunk_dialogue(sid, turns, role):
    """Split one dialogue into windows of at most CHUNK turns with 2 turns of leading context.

    Same as `annotate_nto_moves.chunk_dialogue` except that the coded role is a parameter.
    Both speakers are always rendered into the window, so the student pass still sees the
    tutor's turns as context.

    Args:
        sid: session id.
        turns: the session's turns from `build_session_dialogue` (dicts with `turn` and
            optional `teacher` / `student` text).
        role: "tutor" or "student", the role whose turns are coded.

    Yields:
        ``((sid, set_of_turn_numbers_to_code), user_prompt)`` per window that contains at
        least one turn of the coded role.
    """
    field = "teacher" if role == "tutor" else "student"
    for s in range(0, len(turns), CHUNK):
        lo = max(0, s - 2)
        window = turns[lo:s + CHUNK]
        want = [t["turn"] for t in turns[s:s + CHUNK] if t.get(field)]
        if not want:
            continue
        lines = []
        for t in window:
            if t.get("teacher"):
                lines.append(f"Tutor Turn {t['turn']}: {t['teacher']}")
            if t.get("student"):
                lines.append(f"Student Turn {t['turn']}: {t['student']}")
        user = ("[DIALOGUE CHUNK]\n" + "\n".join(lines) +
                f"\n[END CHUNK]\n\nClassify {role} turns {want}.")
        yield (sid, set(want)), user


def session_order():
    """Return the annotation order of the training sessions.

    One session per learning objective first, then the rest shuffled with SEED; the same
    order as `annotate_nto_moves.py`.

    Returns:
        List of session ids.
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
    """Annotate this shard's sessions block by block, resuming from existing output.

    Returns:
        None. Appends rows to ``<ANNO_OUTDIR>/moves_v2_{arm}_{role}_shard{i}.csv`` per role;
        exits with status 2 when the parse-failure rate exceeds 5% or the vocabulary audit
        fails.
    """
    from vllm import LLM
    from transformers import AutoTokenizer

    os.makedirs(OUTDIR, exist_ok=True)
    out = {r: f"{OUTDIR}/moves_v2_{ARM}_{r}_shard{SHARD_I}.csv" for r in ROLES}
    specs = {r: pass_spec(r) for r in ROLES}

    # Resume: a session counts as done only when every requested role has it, so an
    # interrupted block is redone for both pools rather than half-skipped.
    done = None
    for r in ROLES:
        have = set(pd.read_csv(out[r])["session_id"].unique()) if os.path.exists(out[r]) else set()
        done = have if done is None else (done & have)
    order = [s for i, s in enumerate(session_order()) if i % SHARD_N == SHARD_I]
    if N_SESSIONS:
        order = order[:N_SESSIONS]
    order = [s for s in order if s not in (done or set())]
    print(f"[v2/{ARM} shard {SHARD_I}/{SHARD_N}] model={MODEL} roles={ROLES} "
          f"sessions to annotate: {len(order)}", flush=True)
    if not order:
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, tensor_parallel_size=TP,
              dtype=os.environ.get("ANNO_DTYPE", "bfloat16"),
              max_model_len=16384, gpu_memory_utilization=0.92,
              **_aj.engine_kwargs())
    sp = {r: _aj.make_sampling_params(specs[r][1], MAXTOK) for r in ROLES}
    audit = {r: _aj.VocabAudit(specs[r][2]) for r in ROLES}

    n_chunks = {r: 0 for r in ROLES}
    n_fail = {r: 0 for r in ROLES}
    for bstart in range(0, len(order), BLOCK):
        sids = order[bstart:bstart + BLOCK]
        dialogues = {}
        for sid in sids:
            path = f"{DATA}/train_transcripts/{sid}.csv"
            if not os.path.exists(path):
                continue
            turns = build_session_dialogue(pd.read_csv(path))
            if turns:
                dialogues[sid] = turns
        # Both pools go into one generate() call, with a per-request SamplingParams (vLLM
        # accepts Sequence[SamplingParams]) so the two schemas share a batch. An interrupted
        # block therefore never leaves tutor labels without their student twin. The batch is
        # also twice as wide.
        prompts, metas, sps = [], [], []
        for role in ROLES:
            system = specs[role][0]
            for sid, turns in dialogues.items():
                for meta, user in chunk_dialogue(sid, turns, role):
                    prompts.append(tok.apply_chat_template(
                        [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False))
                    metas.append((role,) + meta)
                    sps.append(sp[role])
        if prompts:
            outs = llm.generate(prompts, sps)
        else:
            outs = []
        rows_by_role = {r: [] for r in ROLES}
        for (role, sid, want), o in zip(metas, outs):
            aux = specs[role][3]
            n_chunks[role] += 1
            obj = _aj.extract_json(o.outputs[0].text)
            try:
                if obj is None:
                    raise ValueError("unparseable")
                for lab in (obj if isinstance(obj, list) else obj["labels"]):
                    t = int(lab["turn"])
                    if t not in want:
                        continue
                    keep = audit[role].check([lab["code"]])
                    if not keep:
                        continue              # unknown primary => no usable row
                    c2 = lab.get("code_2")
                    c2 = "" if not c2 or c2 == NONE_SENTINEL else (audit[role].check([c2])
                                                                   or [""])[0]
                    rows_by_role[role].append({"session_id": sid, "turn": t, "code": keep[0],
                                               "code_2": c2, aux: lab.get(aux, "")})
            except Exception:
                n_fail[role] += 1
        # Write both pools together, so a checkpoint never holds half a session.
        for role, rows in rows_by_role.items():
            if rows:
                pd.DataFrame(rows).to_csv(out[role], mode="a",
                                          header=not os.path.exists(out[role]), index=False)
        done_n = min(bstart + BLOCK, len(order))
        print("  sessions {}/{}  ".format(done_n, len(order)) +
              "  ".join(f"{r}: chunks {n_chunks[r]} fail {n_fail[r]}" for r in ROLES), flush=True)

    bad = 0
    for role in ROLES:
        rate = n_fail[role] / max(1, n_chunks[role])
        print(f"DONE {role} shard {SHARD_I}/{SHARD_N}: chunks {n_chunks[role]}, "
              f"parse_fail {n_fail[role]} ({rate:.2%}) -> {out[role]}", flush=True)
        bad += audit[role].report(f" [v2 {role} shard {SHARD_I}]") + (rate > 0.05)
    # extract_json returns None on truncation rather than guessing, so a cut chunk is counted
    # as a failure. The non-zero exit keeps a degraded run from reaching the distiller.
    if bad:
        print("STOP: parse failures above 5% or systematic vocabulary drift; do not use this run.",
              flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
