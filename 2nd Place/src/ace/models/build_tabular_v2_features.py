#!/usr/bin/env python
r"""Build the structured transcript features of the tabular_v2 model (training stage 4).

Writes session_features.csv (one row per session), response_lexical.csv (one row per response),
with --with-bge response_student_bge.csv (student-side BGE coverage, one row per response), and
config.json.

    python -m ace.models.build_tabular_v2_features --data-dir data \
        --transcripts-dir data/train_transcripts --out outputs/features/tabular_v2 \
        --with-bge --bge-model models/hf/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ace.tabular_v2_features import (
    build_response_lexical_features,
    build_session_feature_table,
    build_student_bge_coverage,
)


def load_bge_encoder(name_or_path: str, device: str | None = None):
    """A loaded SentenceTransformer for build_student_bge_coverage.

    That helper (shared with inference/main.py, so it must stay byte-identical) treats any object
    with an `encode` attribute as a loaded encoder, and a str has str.encode, so a model path passed
    straight through would fail. Load it here with the helper's own device rule.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(name_or_path, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="directory containing train_features.csv")
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--out", default="outputs/features/tabular_v2")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-sessions", type=int, default=1000,
                    help="write progress every N sessions; interrupted runs can resume")
    ap.add_argument("--with-bge", action="store_true")
    ap.add_argument("--bge-model", default="BAAI/bge-base-en-v1.5",
                    help="HF id or local model directory; use a local path on offline machines")
    ap.add_argument("--bge-batch-size", type=int, default=256)
    ap.add_argument("--session-batch-size", type=int, default=96)
    ap.add_argument("--device", default=None, help="cuda, cpu, or omit for auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    feats = pd.read_csv(data_dir / "train_features.csv")
    feats["learning_objective"] = feats["learning_objective"].fillna("").astype(str).str.strip()
    session_ids = sorted(feats["session_id"].astype(str).unique())

    session_path = out / "session_features.csv"
    session_part = out / "session_features.part.csv"
    if args.force:
        session_path.unlink(missing_ok=True)
        session_part.unlink(missing_ok=True)
    if not session_path.exists():
        completed = set()
        if session_part.exists():
            completed = set(pd.read_csv(session_part, usecols=["session_id"])["session_id"].astype(str))
            print(f"[tabv2-features] resume session extraction after {len(completed):,} sessions")
        remaining = [sid for sid in session_ids if sid not in completed]
        for i in range(0, len(remaining), args.chunk_sessions):
            chunk = remaining[i:i + args.chunk_sessions]
            block = build_session_feature_table(
                args.transcripts_dir, session_ids=chunk, workers=args.workers
            )
            block.to_csv(session_part, mode="a", header=not session_part.exists(), index=False)
            print(f"[tabv2-features] session extraction {len(completed) + i + len(chunk):,}/"
                  f"{len(session_ids):,}", flush=True)
        session_df = (pd.read_csv(session_part).drop_duplicates("session_id", keep="last")
                      .sort_values("session_id"))
        session_df.to_csv(session_path, index=False)
        session_part.unlink(missing_ok=True)
    else:
        print(f"[tabv2-features] reuse {session_path}")
        session_df = pd.read_csv(session_path)
    if session_df["session_id"].nunique() != len(session_ids):
        raise RuntimeError("session feature count does not match train_features")

    lexical_path = out / "response_lexical.csv"
    lexical_part = out / "response_lexical.part.csv"
    if args.force:
        lexical_path.unlink(missing_ok=True)
        lexical_part.unlink(missing_ok=True)
    if not lexical_path.exists():
        completed = set()
        if lexical_part.exists():
            completed = set(pd.read_csv(lexical_part, usecols=["response_id"])["response_id"].astype(str))
            print(f"[tabv2-features] resume lexical extraction after {len(completed):,} responses")
        remaining_sids = [sid for sid in session_ids
                          if not set(feats.loc[feats["session_id"].eq(sid), "response_id"].astype(str)).issubset(completed)]
        done_before = len(completed)
        for i in range(0, len(remaining_sids), args.chunk_sessions):
            chunk = remaining_sids[i:i + args.chunk_sessions]
            block_feats = feats[feats["session_id"].isin(chunk)]
            block = build_response_lexical_features(
                block_feats, args.transcripts_dir, workers=args.workers
            )
            block.to_csv(lexical_part, mode="a", header=not lexical_part.exists(), index=False)
            print(f"[tabv2-features] lexical extraction at least "
                  f"{done_before + sum(feats["session_id"].isin(remaining_sids[:i + len(chunk)])):,}/"
                  f"{len(feats):,} responses", flush=True)
        lexical_df = (pd.read_csv(lexical_part).drop_duplicates("response_id", keep="last")
                      .sort_values("response_id"))
        lexical_df.to_csv(lexical_path, index=False)
        lexical_part.unlink(missing_ok=True)
    else:
        print(f"[tabv2-features] reuse {lexical_path}")
        lexical_df = pd.read_csv(lexical_path)
    if lexical_df["response_id"].nunique() != len(feats):
        raise RuntimeError("response lexical feature count does not match train_features")

    bge_path = out / "response_student_bge.csv"
    if args.with_bge:
        if args.force or not bge_path.exists():
            print(f"[tabv2-features] student-only BGE coverage: {len(feats):,} responses")
            bge_df = build_student_bge_coverage(
                feats,
                args.transcripts_dir,
                model_name_or_path=load_bge_encoder(args.bge_model, args.device),
                batch_size=args.bge_batch_size,
                session_batch_size=args.session_batch_size,
                device=args.device,
            )
            bge_df.to_csv(bge_path, index=False)
        else:
            print(f"[tabv2-features] reuse {bge_path}")

    config = {
        "data_dir": str(data_dir),
        "transcripts_dir": str(Path(args.transcripts_dir).expanduser()),
        "n_sessions": len(session_ids),
        "n_responses": len(feats),
        "with_bge": bool(args.with_bge),
        "bge_model": args.bge_model if args.with_bge else None,
        "session_columns": list(session_df.columns),
        "lexical_columns": list(lexical_df.columns),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[tabv2-features] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
