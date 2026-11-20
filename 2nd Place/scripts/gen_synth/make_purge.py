#!/usr/bin/env python
r"""Per-fold near-twin purge list for synthetic training rows (CV hygiene).

A synthetic row whose objective is a near-twin (BGE cosine > --threshold) of an objective in a
fold's validation set would leak twin signal into that fold's CV score. Writes the
(response_id, fold) pairs train_lora.py drops via --synth-purge-csv. Only for training with
synthetic rows; no shipped model was trained on them.

    python scripts/gen_synth/make_purge.py --synth-dir outputs/synth/pilot --data-dir data \
        --folds outputs/eda/cv_folds_skill.csv --out outputs/synth/purge.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", required=True,
                    help="dir with the synth features.csv (e.g. outputs/synth/pilot)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--folds", default="outputs/eda/cv_folds_skill.csv")
    ap.add_argument("--bge", default="inference/assets/bge-base-en-v1.5",
                    help="bge model path or hub name (BAAI/bge-base-en-v1.5)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="near-twin cosine cut (0.85 = the sibling band, aggressive)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data_dir).expanduser()
    feats = pd.read_csv(data / "train_features.csv")
    feats["learning_objective"] = feats["learning_objective"].str.strip()
    df = feats.merge(pd.read_csv(args.folds)[["response_id", "fold"]],
                     on="response_id", validate="1:1")
    real = (df.drop_duplicates("learning_objective_id")
              .sort_values("learning_objective_id").reset_index(drop=True))
    real_ix = {k: i for i, k in enumerate(real["learning_objective_id"])}

    syn = pd.read_csv(Path(args.synth_dir).expanduser() / "features.csv")
    syn["learning_objective"] = syn["learning_objective"].str.strip()
    syn_lo = syn.drop_duplicates("learning_objective_id").reset_index(drop=True)
    print(f"[purge] {len(syn):,} synth rows / {len(syn_lo)} objectives "
          f"vs {len(real)} real objectives, threshold {args.threshold}")

    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.bge,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    emb_real = model.encode(real["learning_objective"].tolist(), batch_size=64,
                            normalize_embeddings=True, convert_to_numpy=True)
    emb_syn = model.encode(syn_lo["learning_objective"].tolist(), batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True)
    S = emb_syn @ emb_real.T
    syn_obj_ix = syn["learning_objective_id"].map(
        {k: i for i, k in enumerate(syn_lo["learning_objective_id"])}).to_numpy()

    rows = []
    for f in sorted(df["fold"].unique()):
        val_ids = df.loc[df["fold"] == f, "learning_objective_id"].unique()
        mx = S[:, [real_ix[i] for i in val_ids]].max(axis=1)[syn_obj_ix]
        hit = mx > args.threshold
        print(f"[purge] fold {f}: drop {int(hit.sum()):,}/{len(syn):,} rows "
              f"({hit.mean():.1%})")
        rows.append(pd.DataFrame({"response_id": syn.loc[hit, "response_id"],
                                  "fold": f, "max_sim": np.round(mx[hit], 4)}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"[purge] wrote {len(out):,} (response_id, fold) pairs -> {args.out}")


if __name__ == "__main__":
    main()