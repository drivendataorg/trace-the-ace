"""Self-annotate Trace the Ace transcripts with per-turn correctness labels (vLLM version).

Same design as annotate_traceace_qwen32b.py (chunks of at most ANNO_CHUNK_TURNS turns, per-turn
summary chain-of-thought, extract_result parsing, resume-safe append CSV), but generation runs
through vLLM (continuous batching and paged KV cache) instead of transformers.generate, with the
same prompts and greedy decoding.

Modes:
  Parity test: ANNO_PARITY=1 re-annotates N_PARITY
    sessions from the ANNO_REF CSV and prints the label agreement with it (PASS at >= 0.90).
  Production: shard over sessions with ANNO_SHARD="i/N" (default 0/1). Each shard writes its
    own CSV (results/traceace_turn_labels_vllm_shard<i>.csv) so appends never race; merge the
    shards afterwards. Sessions already in ANNO_REF or in the shard's own CSV are skipped; step
    02a sets ANNO_REF to a file that does not exist, so only the shard's own CSV is checked.

Unparseable replies are logged to results/anno_parse_fails.jsonl.

Environment: ANNO_MODEL (Qwen/Qwen3-32B), ANNO_TP=1, ANNO_N_SESSIONS (0 = all),
ANNO_CHUNK_TURNS=30, ANNO_MAXTOK=2048, ANNO_SHARD="0/1", ANNO_PARITY, ANNO_N_PARITY=20,
ANNO_REF=results/traceace_turn_labels.csv, plus the anno_json.engine_kwargs variables.
"""
import os, sys, random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.annotate import extract_result
from dialogue_kt.prompting import ANNO_CORRECTNESS_SYSTEM_PROMPT, TRACEACE_DIALOGUE_DESC
from build_traceace_kt import build_session_dialogue, find_one, MAGNIFICAT
from annotate_traceace_qwen32b import chunk_prompts  # identical chunking/prompt text

MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
TP = int(os.environ.get("ANNO_TP", "1"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "30"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "2048"))
SHARD_I, SHARD_N = (int(x) for x in os.environ.get("ANNO_SHARD", "0/1").split("/"))
PARITY = os.environ.get("ANNO_PARITY", "0") == "1"
N_PARITY = int(os.environ.get("ANNO_N_PARITY", "20"))
N_SESSIONS = int(os.environ.get("ANNO_N_SESSIONS", "0"))  # 0 = no cap
REF = os.environ.get("ANNO_REF", "results/traceace_turn_labels.csv")
OUT = f"results/traceace_turn_labels_vllm_shard{SHARD_I}.csv"
BLOCK = 60  # sessions per generate() call (~240 chunks in flight)
DATA = f"{MAGNIFICAT}/data"
SEED = 221


def session_order():
    """Order all training sessions as the HF script does: one per objective first, then the rest.

    Returns:
        A list of session ids (seed 221) covering every training session.
    """
    feats = pd.read_csv(find_one(f"{DATA}/train_features*.csv"))
    rng = random.Random(SEED)
    first = feats.groupby("learning_objective_id", group_keys=False).head(1)
    sids = list(dict.fromkeys(first["session_id"]))
    rest = [s for s in feats["session_id"].unique() if s not in set(sids)]
    rng.shuffle(rest)
    return sids + rest


def main():
    """Run the parity test or annotate this shard's sessions.

    Returns:
        None. In production mode appends ``session_id, turn, label`` rows to OUT; in parity
        mode only prints the agreement with ANNO_REF.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    ref = pd.read_csv(REF) if os.path.exists(REF) else pd.DataFrame(columns=["session_id", "turn", "label"])
    order = session_order()

    if PARITY:
        sids = [s for s in order if s in set(ref.session_id)][:N_PARITY]
        out_rows_only = True
        print(f"PARITY mode: re-annotating {len(sids)} HF-annotated sessions", flush=True)
    else:
        done_ref = set(ref.session_id)
        done_own = set(pd.read_csv(OUT).session_id) if os.path.exists(OUT) else set()
        sids = [s for i, s in enumerate(order) if i % SHARD_N == SHARD_I
                and s not in done_ref and s not in done_own]
        if N_SESSIONS:
            sids = sids[:N_SESSIONS]
        out_rows_only = False
        print(f"PRODUCTION shard {SHARD_I}/{SHARD_N}: {len(sids)} sessions to annotate -> {OUT}", flush=True)

    system = ANNO_CORRECTNESS_SYSTEM_PROMPT.format(desc=TRACEACE_DIALOGUE_DESC)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # Pipeline parallelism and prefix caching come from anno_json.engine_kwargs. The
    # correctness system prompt is identical across every chunk, so prefix caching skips its
    # repeated prefill. This annotator uses plain SamplingParams (no grammar), so it needs none
    # of the SO_OFF handling of the NTO and student annotators.
    import sys as _s
    _s.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import anno_json as _aj
    llm = LLM(model=MODEL, dtype="auto", tensor_parallel_size=TP, max_model_len=8192,
              gpu_memory_utilization=0.92, enforce_eager=False, **_aj.engine_kwargs())
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    header_needed = not (not out_rows_only and os.path.exists(OUT))
    all_rows, n_parse_fail, n_chunks = [], 0, 0

    for bstart in range(0, len(sids), BLOCK):
        block = sids[bstart:bstart + BLOCK]
        metas, prompts = [], []
        for sid in block:
            path = f"{DATA}/train_transcripts/{sid}.csv"
            if not os.path.exists(path):
                continue
            turns = build_session_dialogue(pd.read_csv(path))
            if not turns:
                continue
            for meta, up in chunk_prompts(sid, turns):
                metas.append(meta)
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": up}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
        n_chunks += len(prompts)
        outs = llm.generate(prompts, sp)
        rows = []
        for (sid, start, k), o in zip(metas, outs):
            pred = extract_result(o.outputs[0].text)
            if not isinstance(pred, dict):
                # `extract_result` only accepts fenced JSON (```json {...}```) or an unfenced
                # `result = {...}`; a bare `{...}` reply is parsed only by this fallback.
                # `extract_json` is the balanced-brace, string-aware scanner used by the NTO and
                # student annotators; it returns None on truncation instead of half parsing.
                pred = _aj.extract_json(o.outputs[0].text)
                if isinstance(pred, dict) and "result" in pred:
                    pred = pred["result"]
            if not isinstance(pred, dict):
                n_parse_fail += 1
                # record why: finish_reason ('length' = truncated) and the output tail
                import json as _json
                with open("results/anno_parse_fails.jsonl", "a") as fh:
                    fh.write(_json.dumps({
                        "session_id": sid, "start": start, "k": k,
                        "finish_reason": str(o.outputs[0].finish_reason),
                        "n_out_tokens": len(o.outputs[0].token_ids),
                        "tail": o.outputs[0].text[-400:]}) + "\n")
                continue
            for key, val in pred.items():
                try:
                    local = int(str(key).lower().replace("turn", "").strip())
                except ValueError:
                    continue
                if 1 <= local <= k and str(val).lower().strip() in ("true", "false", "na"):
                    rows.append({"session_id": sid, "turn": start + local,
                                 "label": str(val).lower().strip()})
        if out_rows_only:
            all_rows.extend(rows)
        elif rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=header_needed, index=False)
            header_needed = False
        print(f"  sessions {min(bstart + BLOCK, len(sids))}/{len(sids)}  chunks {n_chunks}  "
              f"parse_fail {n_parse_fail}", flush=True)

    if PARITY:
        new = pd.DataFrame(all_rows)
        m = new.merge(ref, on=["session_id", "turn"], suffixes=("_vllm", "_hf"))
        ag = (m.label_vllm == m.label_hf).mean()
        binm = m[m.label_hf.isin(["true", "false"]) & m.label_vllm.isin(["true", "false"])]
        print(f"PARITY: {len(m)} shared turns | 3-way agreement {ag:.3f} | "
              f"binary agreement {(binm.label_vllm == binm.label_hf).mean():.3f} (n={len(binm)}) | "
              f"parse_fail {n_parse_fail}/{n_chunks}")
        print("PARITY " + ("PASS" if ag >= 0.90 else "below 0.90: inspect the disagreements before the production run"))
    else:
        print(f"DONE shard {SHARD_I}/{SHARD_N}: chunks {n_chunks}, parse_fail {n_parse_fail} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
