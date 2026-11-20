#!/usr/bin/env python
r"""Stage 5: how close is the synthetic holdout to the real distribution? Writes <synth-dir>/QC.md.

1. transcript-statistic quantiles, synthetic vs real, through the parser the models use
   (ace.transcripts);
2. a real-vs-synthetic TF-IDF logistic-regression discriminator, 5-fold AUC (0.5 = indistinguishable);
3. label sanity: realised prior, band mix, p_soft calibration.

    PYTHONPATH=src python scripts/gen_synth/qc.py --synth-dir outputs/synth/pilot \
        --real-stats outputs/eda/transcript_stats.csv --transcripts-dir data/train_transcripts
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ace.transcripts import parse_transcript

QCOLS = ["n_utt", "w_total", "dur_s", "n_background", "n_unclear", "tutor_q",
         "student_q", "w_tutor", "w_student", "wpu_tutor", "wpu_student"]


def qtable(df, cols, qs=(0.25, 0.5, 0.75)):
    return {c: [round(float(df[c].quantile(x)), 1) for x in qs] for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", default="outputs/synth/pilot")
    ap.add_argument("--real-stats", default="outputs/eda/transcript_stats.csv")
    ap.add_argument("--transcripts-dir", required=True,
                    help="real transcripts (for the discriminator sample)")
    ap.add_argument("--n-disc", type=int, default=2000,
                    help="real sessions sampled for the discriminator")
    args = ap.parse_args()
    sdir = Path(args.synth_dir).expanduser()
    rng = np.random.default_rng(42)

    syn_files = sorted(glob.glob(str(sdir / "transcripts" / "*.csv")))
    parsed = [parse_transcript(p) for p in syn_files]
    syn = pd.DataFrame([st for _, st, _, _ in parsed])
    syn_text = {sid: tx for sid, _, tx, _ in parsed}
    real = pd.read_csv(args.real_stats)
    # words per utterance by role: the statistic generated text gets most wrong, so it is
    # watched directly rather than through w_total
    for df_ in (syn, real):
        df_["wpu_tutor"] = df_["w_tutor"] / df_["n_tutor"].clip(lower=1)
        df_["wpu_student"] = df_["w_student"] / df_["n_student"].clip(lower=1)

    R = [f"# Synthetic pilot QC — {len(syn)} sessions\n",
         "## 1. Stat quantiles (p25 / p50 / p75), synthetic vs real\n",
         "| stat | synthetic | real |", "|---|---|---|"]
    rq, sq = qtable(real, QCOLS), qtable(syn, QCOLS)
    for c in QCOLS:
        R.append(f"| {c} | {' / '.join(map(str, sq[c]))} | {' / '.join(map(str, rq[c]))} |")

    # ---- discriminator
    real_sample = rng.choice(glob.glob(str(Path(args.transcripts_dir).expanduser()
                                            / "*.csv")), args.n_disc, replace=False)
    real_text = [parse_transcript(p)[2] for p in real_sample]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X_txt = real_text + list(syn_text.values())
    y = np.r_[np.zeros(len(real_text)), np.ones(len(syn_text))]
    vec_ = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                           sublinear_tf=True)
    X = vec_.fit_transform(X_txt)
    auc = cross_val_score(LogisticRegression(C=1.0, max_iter=2000), X, y,
                          cv=5, scoring="roc_auc")
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    terms = np.array(vec_.get_feature_names_out())
    o = np.argsort(clf.coef_[0])
    R += ["\n## 2. Real-vs-synthetic TF-IDF discriminator\n",
          f"5-fold AUC: **{auc.mean():.3f}** ± {auc.std():.3f} "
          "(1.0 = trivially separable, 0.5 = perfect camouflage; expect high on a "
          "first pilot — track the trend)\n",
          f"- top synthetic tells: {', '.join(terms[o[-25:]][::-1])}",
          f"- top real tells: {', '.join(terms[o[:25]])}\n"]

    labels = pd.read_csv(sdir / "labels.csv")
    meta = [json.loads(ln) for ln in open(sdir / "meta.jsonl", encoding="utf-8")]
    bands = pd.Series([b for m in meta for b in m["bands"]]).value_counts(normalize=True)
    bins = pd.cut(labels["p_soft"], [0, .25, .45, .65, .85, 1.0])
    cal = labels.groupby(bins, observed=True)["is_correct"].agg(["mean", "size"])
    R += ["\n## 3. Labels\n",
          f"- realized prior: **{labels['is_correct'].mean():.4f}** (target 0.7025)",
          f"- novelty bands: {bands.round(3).to_dict()}",
          "- p_soft calibration (bin -> realized rate / n):",
          cal.round(3).to_markdown(), ""]

    (sdir / "QC.md").write_text("\n".join(R), encoding="utf-8")
    print("\n".join(R))
    print(f"\nwrote {sdir / 'QC.md'}")


if __name__ == "__main__":
    main()