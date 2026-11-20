#!/usr/bin/env python
"""Build a resumable BGE semantic index for objective-aware LLM prompts.

Each output row contains only the top transcript line spans for one response. Labels
are never loaded. Candidate passage embeddings are shared across objectives belonging
to the same session. The resulting JSONL is consumed by both train_lora.py and
score_vllm.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ace.llm.objective_context import (
    BGE_QUERY_PREFIX, INDEX_VERSION, candidate_windows, context_meta_path,
    load_context_index, repair_trailing_partial_jsonl, select_ranked_windows,
)
from ace.llm.serialize import read_dialogue


def _read_one(payload):
    sid, path, serialize, window_words, stride_words, min_words = payload
    lines = read_dialogue(path, version=serialize)
    wins = candidate_windows(lines, window_words=window_words,
                             stride_words=stride_words, min_words=min_words)
    texts = ["\n".join(lines[a:b]) for a, b in wins]
    return sid, wins, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="train_features.csv or test_features.csv")
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--out", required=True, help="JSONL context index")
    ap.add_argument("--bge-model", required=True)
    ap.add_argument("--serialize", type=int, default=1, choices=[1, 2])
    ap.add_argument("--window-words", type=int, default=240)
    ap.add_argument("--stride-words", type=int, default=120)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-overlap", type=float, default=0.55)
    ap.add_argument("--session-batch-size", type=int, default=96)
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = args.device or "cpu"

    features_path = Path(args.features).expanduser()
    tdir = Path(args.transcripts_dir).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = context_meta_path(out)

    # A content fingerprint prevents accidentally appending test rows to a train index
    # (or vice versa) even when the same output path is reused on another machine.
    digest = hashlib.sha256()
    with features_path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    meta = {
        "version": INDEX_VERSION,
        "features_name": features_path.name,
        "features_size": features_path.stat().st_size,
        "features_sha256": digest.hexdigest(),
        "bge_model": str(Path(args.bge_model).expanduser()),
        "serialize": args.serialize,
        "window_words": args.window_words,
        "stride_words": args.stride_words,
        "min_words": args.min_words,
        "top_k": args.top_k,
        "max_overlap": args.max_overlap,
    }

    if args.overwrite:
        out.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != meta:
            raise SystemExit(f"existing context index has incompatible metadata:\n{old}\n!=\n{meta}")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))

    df = pd.read_csv(features_path, usecols=["response_id", "session_id", "learning_objective"])
    for c in ("response_id", "session_id", "learning_objective"):
        df[c] = df[c].fillna("").astype(str)
    if df["response_id"].duplicated().any():
        raise SystemExit("features contains duplicate response_id values")

    done: set[str] = set()
    if out.exists() and out.stat().st_size:
        if repair_trailing_partial_jsonl(out):
            print(f"[context] repaired an incomplete trailing JSONL fragment in {out}")
        done = set(load_context_index(out))
        print(f"[context] resume: {len(done):,} response rows already present")
    remaining = df[~df["response_id"].isin(done)].copy()
    if remaining.empty:
        print(f"[context] complete: {out}")
        return

    model = SentenceTransformer(str(Path(args.bge_model).expanduser()), device=device)
    objective_texts = remaining["learning_objective"].drop_duplicates().tolist()
    q_emb = model.encode(
        [BGE_QUERY_PREFIX + x for x in objective_texts],
        batch_size=min(128, args.encode_batch_size), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    q_ix = {x: i for i, x in enumerate(objective_texts)}
    by_session = {str(sid): g.copy() for sid, g in remaining.groupby("session_id", sort=True)}
    sids = list(by_session)
    workers = max(1, min(args.workers, os.cpu_count() or 4))

    with out.open("a", encoding="utf-8") as writer:
        for b0 in range(0, len(sids), args.session_batch_size):
            batch_sids = sids[b0:b0 + args.session_batch_size]
            payloads = [
                (sid, str(tdir / f"{sid}.csv"), args.serialize, args.window_words,
                 args.stride_words, args.min_words)
                for sid in batch_sids
            ]
            if workers == 1:
                parsed = [_read_one(x) for x in payloads]
            else:
                with Pool(workers) as pool:
                    parsed = pool.map(_read_one, payloads, chunksize=16)
            passages: list[str] = []
            bounds: dict[str, tuple[int, int, list[tuple[int, int]]]] = {}
            for sid, wins, texts in parsed:
                a = len(passages)
                passages.extend(texts)
                bounds[sid] = (a, len(passages), wins)
            if passages:
                p_emb = model.encode(
                    passages, batch_size=args.encode_batch_size,
                    normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=False,
                )
            else:
                p_emb = np.empty((0, q_emb.shape[1]), dtype=np.float32)

            for sid in batch_sids:
                a, b, wins = bounds[sid]
                g = by_session[sid]
                for row in g.itertuples(index=False):
                    if b > a:
                        scores = p_emb[a:b] @ q_emb[q_ix[row.learning_objective]]
                        selected = select_ranked_windows(
                            wins, scores, top_k=args.top_k, max_overlap=args.max_overlap)
                    else:
                        selected = []
                    payload = {
                        "response_id": str(row.response_id),
                        "session_id": sid,
                        "serialize": args.serialize,
                        "windows": [[w.start, w.end, round(float(w.score), 7)] for w in selected],
                    }
                    writer.write(json.dumps(payload, separators=(",", ":")) + "\n")
            writer.flush()
            print(f"[context] sessions {min(b0 + args.session_batch_size, len(sids)):,}/"
                  f"{len(sids):,}", flush=True)
    print(f"[context] wrote {out}")


if __name__ == "__main__":
    main()
