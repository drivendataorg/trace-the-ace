#!/usr/bin/env python
"""Annotate the interview-codebook labels with an LLM and keep the raw per-utterance records.

The codebook is described in `src/training/docs/29_interviews_and_labels.md`. Two passes
(prompts in `magnificat.doc29_prompts`):
  doc29_student: per student utterance, reasoning / confidence / affect
  doc29_tutor:   per tutor utterance, one of the 7 interview-codebook moves, or NONE

The output has one row per annotated utterance with every field, so both the oracle features
and the distillation targets are built from the same file.

Resumable by chunk: each chunk of sessions is written to its own parquet file as soon as it
finishes, so a dropped connection or an expiring job costs one chunk. A re-run skips the chunks
already on disk.

Environment: PP and TP (pipeline / tensor parallel size, default 1), ANNO_DTYPE (default
bfloat16) and VLLM_MAXLEN (default 8192) configure vLLM.

In:  data/train_features.csv, data/train_transcripts/
Out: <out-dir>/chunks/chunk_NNN.parquet, <out-dir>/doc29_raw_records_14b.parquet

Usage (reproduce.py step 02d, vLLM env, from build/workspace/; ~75 min on 1x B200):
  PYTHONPATH=src python scripts/annotate_doc29_labels.py --qwen Qwen/Qwen2.5-14B-Instruct \
    --transcripts data/train_transcripts --out-dir artifacts/doc29_records_14b --sessions 4000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from magnificat import data  # noqa: E402
from magnificat import doc29_prompts as D29  # noqa: E402
from magnificat import sandpiper_live_infer as sli  # noqa: E402


def _parse_multi(text: str, expected_ids: list[int], fields: dict) -> list[dict]:
    """Parse a JSON array of multi-field records from one generation.

    `sli._parse_output` assumes a single `code` key; these passes carry up to three
    fields, so this is the multi-field version with the same tolerance: a dict wrapper
    around the list is accepted, a missing or non-integer utterance id falls back to the
    positional id, and a record with an out-of-vocabulary value is dropped.

    Args:
        text: the model's raw output text.
        expected_ids: utterance ids the prompt asked about, in prompt order.
        fields: field name -> set of allowed values.

    Returns:
        One dict per valid record: `utterance_id` plus one key per field. Empty when the
        text is not valid JSON.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        lists = [v for v in parsed.values() if isinstance(v, list)]
        items = lists[0] if lists else [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        uid = item.get("utterance_id")
        try:
            uid = int(uid)
        except (TypeError, ValueError):
            uid = expected_ids[idx] if idx < len(expected_ids) else None
        if uid is None:
            continue
        rec = {"utterance_id": uid}
        ok = True
        for fname, allowed in fields.items():
            val = item.get(fname)
            if not isinstance(val, str) or val not in allowed:
                ok = False
                break
            rec[fname] = val
        if ok:
            out.append(rec)
    return out


def _run_pass(sess: list[str], store, tok, llm, name: str, cfg: dict) -> tuple[list[dict], dict]:
    """Annotate one pass over a list of sessions and count what was lost.

    The counts matter: `_parse_multi` returns only the records that parsed, so a batch whose
    JSON array is cut off at ``max_tokens`` yields labels for the first utterances and drops
    the tail, which looks the same as "the model found nothing". ``n_trunc`` counts
    generations that stopped on the length cap; ``n_expected`` and ``n_labelled`` bound the
    loss end to end. These are training-set counts (the competition rules forbid logging
    test-set aggregates, not these).

    Args:
        sess: session ids to annotate.
        store: `magnificat.data.TranscriptStore` holding the transcripts.
        tok: Hugging Face tokenizer of the annotating model.
        llm: vLLM `LLM` instance.
        name: pass name, written to the `annotation_type` column.
        cfg: pass configuration from `doc29_prompts.DOC29_PASSES` (role, prompt, schema,
            fields, batch_size, max_tokens, max_prompt_tokens).

    Returns:
        ``(records, stats)``: one dict per labelled utterance (session_id, utterance_id,
        annotation_type, position_frac and the pass's fields), and the counters
        n_prompts, n_trunc, n_expected, n_labelled, n_empty.
    """
    metas, texts = sli.build_prompts_whole(
        sess, store, cfg["batch_size"], tok, cfg["role"], cfg["prompt"],
        max_prompt_tokens=cfg["max_prompt_tokens"])
    stats = {"n_prompts": 0, "n_trunc": 0, "n_expected": 0, "n_labelled": 0, "n_empty": 0}
    if not texts:
        return [], stats
    sp = sli._make_sampling_params(cfg["schema"], max_tokens=cfg["max_tokens"])

    records: list[dict] = []
    chunk = 1000
    for lo in range(0, len(texts), chunk):
        hi = min(lo + chunk, len(texts))
        outs = llm.generate(texts[lo:hi], sp, use_tqdm=False)
        for meta, out in zip(metas[lo:hi], outs):
            o = out.outputs[0]
            anns = _parse_multi(o.text, meta["expected_ids"], cfg["fields"])
            stats["n_prompts"] += 1
            stats["n_trunc"] += int(getattr(o, "finish_reason", None) == "length")
            stats["n_expected"] += len(meta["expected_ids"])
            stats["n_labelled"] += len(anns)
            stats["n_empty"] += int(not anns)
            pos = meta.get("position_fracs", [])
            for j, ann in enumerate(anns):
                rec = {
                    "session_id": meta["session_id"],
                    "utterance_id": ann["utterance_id"],
                    "annotation_type": name,
                    "position_frac": pos[j] if j < len(pos) else float("nan"),
                }
                rec.update({k: v for k, v in ann.items() if k != "utterance_id"})
                records.append(rec)
    return records, stats


def main() -> None:
    """Run the selected passes chunk by chunk and merge the chunks.

    Returns:
        None. Writes one parquet per chunk under ``<out-dir>/chunks/`` and the merged
        ``<out-dir>/doc29_raw_records_14b.parquet``, and prints a coverage audit.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sessions", type=int, default=4000)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--passes", default="doc29_student,doc29_tutor")
    args = ap.parse_args()

    want = [p.strip() for p in args.passes.split(",") if p.strip()]
    ck = args.out_dir / "chunks"
    ck.mkdir(parents=True, exist_ok=True)

    train = data.load_train()
    store = data.TranscriptStore(args.transcripts)
    # The first --sessions sessions in train-file order.
    sessions = list(pd.unique(train["session_id"]))[:args.sessions]
    chunks = [sessions[i:i + args.chunk_size]
              for i in range(0, len(sessions), args.chunk_size)]
    print(f"doc29 annotation: {len(sessions):,} sessions in {len(chunks)} chunks "
          f"({args.chunk_size}/chunk) -> {args.out_dir}", flush=True)
    print(f"passes: {want}", flush=True)

    import os  # noqa: PLC0415
    os.environ.setdefault("QA2_PREFIX_CACHING", "1")
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")

    todo = [i for i in range(len(chunks)) if not (ck / f"chunk_{i:03d}.parquet").exists()]
    print(f"{len(chunks) - len(todo)} chunks already on disk; {len(todo)} to do", flush=True)
    total_cov: dict[str, int] = {}

    if todo:
        from transformers import AutoTokenizer  # noqa: PLC0415
        from vllm import LLM  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(args.qwen)
        # PP/TP and dtype come from the environment; the defaults (one GPU, bf16) serve the
        # 14B annotator.
        _pp = int(os.environ.get("PP", "1"))
        _kw = {"pipeline_parallel_size": _pp} if _pp > 1 else {}
        llm = LLM(model=args.qwen, dtype=os.environ.get("ANNO_DTYPE", "bfloat16"),
                  tensor_parallel_size=int(os.environ.get("TP", "1")),
                  gpu_memory_utilization=args.gpu_util,
                  max_model_len=int(os.environ.get("VLLM_MAXLEN", "8192")),
                  enforce_eager=False,
                  attention_config={"backend": "TRITON_ATTN"},
                  enable_prefix_caching=True, **_kw)
        t0 = time.time()
        for n, ci in enumerate(todo, 1):
            sess = chunks[ci]
            ct = time.time()
            recs: list[dict] = []
            cov: dict[str, dict] = {}
            for name in want:
                r, st = _run_pass(sess, store, tok, llm, name, D29.DOC29_PASSES[name])
                recs += r
                cov[name] = st
                for k, v in st.items():
                    total_cov[k] = total_cov.get(k, 0) + v
            # write-then-rename so a kill mid-write cannot leave a torn parquet
            tmp = ck / f".tmp_{ci:03d}.parquet"
            pd.DataFrame(recs).to_parquet(tmp)
            tmp.rename(ck / f"chunk_{ci:03d}.parquet")
            el = time.time() - t0
            print(f"[chunk {n}/{len(todo)} (#{ci})] {len(sess)} sess, {len(recs):,} records "
                  f"in {(time.time()-ct)/60:.1f} min | elapsed {el/60:.1f} min | "
                  f"eta {(el/n*(len(todo)-n))/60:.1f} min", flush=True)
            for name, st in cov.items():
                cvr = st["n_labelled"] / max(1, st["n_expected"])
                warn = "  WARNING: truncated" if st["n_trunc"] else ""
                print(f"    coverage[{name}]: {st['n_labelled']:,}/{st['n_expected']:,} "
                      f"utterances = {cvr:.1%} | {st['n_trunc']}/{st['n_prompts']} prompts hit "
                      f"the length cap | {st['n_empty']} parsed empty{warn}", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(ck.glob("chunk_*.parquet"))]
    if not parts:
        print("no chunks produced")
        return
    out = pd.concat(parts, ignore_index=True)
    dst = args.out_dir / "doc29_raw_records_14b.parquet"
    out.to_parquet(dst)
    print(f"\nwrote {dst}  ({len(out):,} records)")
    print("  by pass:", out.annotation_type.value_counts().to_dict())
    print(f"  sessions covered: {out.session_id.nunique():,}")
    if total_cov:
        cvr = total_cov["n_labelled"] / max(1, total_cov["n_expected"])
        print(f"\nCOVERAGE AUDIT (this run's chunks only): "
              f"{total_cov['n_labelled']:,}/{total_cov['n_expected']:,} utterances = {cvr:.2%}; "
              f"{total_cov['n_trunc']:,}/{total_cov['n_prompts']:,} prompts hit the length cap; "
              f"{total_cov['n_empty']:,} parsed empty.")
        if cvr < 0.95 or total_cov["n_trunc"]:
            print("   WARNING: SILENT LOSS. Raise max_tokens in doc29_prompts.DOC29_PASSES (or "
                  "lower batch_size) and re-run the affected chunks; otherwise the distillation "
                  "targets are biased toward early utterances, a position confound in a block "
                  "whose features compare `_late` with `_rate`.")
        else:
            print("   OK: no truncation detected.")
    for col in ("reasoning", "confidence", "affect", "code"):
        if col in out.columns:
            vc = out[col].value_counts(dropna=True)
            print(f"  {col}: {vc.to_dict()}")


if __name__ == "__main__":
    main()
