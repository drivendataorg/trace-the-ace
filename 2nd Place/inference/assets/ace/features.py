"""Feature building for tabular models.

Assembles the response-level feature frame from the EDA / embedding-pass artifacts:
  - transcript surface statistics   (outputs/eda/transcript_stats.csv)
  - objective<->transcript coverage (outputs/embeddings/coverage.csv)
  - lexical overlap                 (outputs/eda/objective_overlap.csv)
  - embedding topics                (outputs/embeddings/objective_topics_emb.csv)
  - fixed CV folds                  (outputs/eda/cv_folds.csv)

plus the fold-safe kNN-backoff objective target encoding (EDA.md §10.2), which needs
the cached objective embeddings (emb_objectives.npy from the embedding pass).

Competition-rule note: `n_resp` (responses per session) is computable at inference from
`test_features.csv` rows of the *same session* only — within-sample context, not a
test-set aggregate — but it does use sibling test rows, so it sits behind the
`include_session_count` flag for easy ablation/removal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRANSCRIPT_FEATURES = [
    "n_utt", "n_tutor", "n_student", "n_background", "w_total", "w_student", "w_tutor",
    "dur_s", "tutor_q", "student_q", "n_unclear", "n_bracket",
    "student_share_w", "w_per_utt_student", "w_per_utt_tutor", "unclear_share",
    "words_per_sec", "q_per_tutor_utt",
]
OBJECTIVE_FEATURES = ["te_lo", "lo_count", "lo_words", "topic_emb"]
COVERAGE_FEATURES = ["coverage_max", "coverage_top3", "coverage_gap", "lo_overlap"]
SESSION_FEATURES = ["n_resp"]
CATEGORICAL_FEATURES = ["topic_emb"]


def feature_columns(include_session_count: bool = True) -> list[str]:
    cols = OBJECTIVE_FEATURES + COVERAGE_FEATURES + TRANSCRIPT_FEATURES
    if include_session_count:
        cols = cols + SESSION_FEATURES
    return cols


def derive_transcript_columns(ts: pd.DataFrame) -> pd.DataFrame:
    """Derived transcript features from the raw per-session statistics.

    One implementation shared by training (transcript_stats.csv) and inference
    (ace.transcripts.parse_transcript output).
    """
    ts = ts.copy()
    ts["student_share_w"] = ts["w_student"] / ts["w_total"].clip(lower=1)
    ts["w_per_utt_student"] = ts["w_student"] / ts["n_student"].clip(lower=1)
    ts["w_per_utt_tutor"] = ts["w_tutor"] / ts["n_tutor"].clip(lower=1)
    ts["unclear_share"] = ts["n_unclear"] / ts["n_utt"].clip(lower=1)
    ts["words_per_sec"] = ts["w_total"] / ts["dur_s"].clip(lower=1)
    ts["q_per_tutor_utt"] = ts["tutor_q"] / ts["n_tutor"].clip(lower=1)
    return ts


def load_frame(data_dir: str | Path, eda_dir: str | Path, emb_dir: str | Path) -> pd.DataFrame:
    """Response-level frame: ids, label, fold, and all static (fold-independent) features."""
    data, eda, emb = Path(data_dir).expanduser(), Path(eda_dir).expanduser(), Path(emb_dir).expanduser()

    df = pd.read_csv(data / "train_features.csv")
    df["learning_objective"] = df["learning_objective"].str.strip()
    df = df.merge(pd.read_csv(data / "train_labels.csv"), on="response_id", validate="1:1")
    df = df.merge(pd.read_csv(eda / "cv_folds.csv")[["response_id", "fold"]],
                  on="response_id", validate="1:1")

    ts = derive_transcript_columns(pd.read_csv(eda / "transcript_stats.csv"))
    keep = ["session_id"] + [c for c in TRANSCRIPT_FEATURES if c in ts.columns]
    df = df.merge(ts[keep], on="session_id", how="left", validate="m:1")

    cov = pd.read_csv(emb / "coverage.csv")[
        ["response_id", "coverage_max", "coverage_top3"]]
    df = df.merge(cov, on="response_id", how="left", validate="1:1")
    df["coverage_gap"] = df["coverage_max"] - df["coverage_top3"]

    ovl = pd.read_csv(eda / "objective_overlap.csv").drop_duplicates(
        ["session_id", "learning_objective_id"])
    df = df.merge(ovl, on=["session_id", "learning_objective_id"], how="left", validate="m:1")

    topics = pd.read_csv(emb / "objective_topics_emb.csv")[
        ["learning_objective_id", "topic_emb"]]
    df = df.merge(topics, on="learning_objective_id", how="left", validate="m:1")
    df["topic_emb"] = df["topic_emb"].fillna(-1).astype(int)

    df["lo_words"] = df["learning_objective"].str.split().str.len()
    df["n_resp"] = df.groupby("session_id")["response_id"].transform("size")
    return df


class KNNBackoffTE:
    """Objective target encoding with kNN backoff over objective embeddings (EDA.md §10.2).

    rate_i is blended with the similarity- and count-weighted mean rate of the k nearest
    objectives, then with the global prior:
        te_i = (sum_i + a_knn * r_knn_i + a_prior * prior) / (cnt_i + a_knn + a_prior)

    The embedding row order MUST match `lo_ids` (sorted learning_objective_id, as written
    by run_embeddings.py).
    """

    def __init__(self, emb_obj: np.ndarray, lo_ids: list[str],
                 k: int = 10, a_knn: float = 10.0, a_prior: float = 10.0):
        assert len(lo_ids) == emb_obj.shape[0], "embeddings/id count mismatch"
        self.lo_ix = {k_: i for i, k_ in enumerate(lo_ids)}
        self.n = len(lo_ids)
        self.k, self.a_knn, self.a_prior = k, a_knn, a_prior
        S = emb_obj @ emb_obj.T
        np.fill_diagonal(S, -1)
        self.nn_idx = np.argsort(-S, axis=1)[:, :k]
        self.nn_sim = np.take_along_axis(S, self.nn_idx, axis=1).clip(min=0)
        self.te_: np.ndarray | None = None
        self.cnt_: np.ndarray | None = None

    def index_of(self, lo_id: pd.Series) -> np.ndarray:
        return lo_id.map(self.lo_ix).fillna(-1).astype(int).to_numpy()

    def fit(self, lo_idx: np.ndarray, y: np.ndarray) -> "KNNBackoffTE":
        prior = float(y.mean())
        cnt = np.bincount(lo_idx, minlength=self.n).astype(float)
        s = np.bincount(lo_idx, weights=y, minlength=self.n)
        rate = np.divide(s, cnt, out=np.full(self.n, prior), where=cnt > 0)
        w = self.nn_sim * cnt[self.nn_idx]
        r_knn = (w * rate[self.nn_idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        self.te_ = (s + self.a_knn * r_knn + self.a_prior * prior) / (cnt + self.a_knn + self.a_prior)
        self.cnt_ = cnt
        self.prior_ = prior
        return self

    def transform(self, lo_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        te = np.where(lo_idx >= 0, self.te_[lo_idx.clip(min=0)], self.prior_)
        cnt = np.where(lo_idx >= 0, self.cnt_[lo_idx.clip(min=0)], 0.0)
        return te, cnt

    def oof(self, lo_idx: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fold-safe encoding: for each fold, fit on the other folds only."""
        te = np.full(len(y), np.nan)
        cnt = np.full(len(y), np.nan)
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            self.fit(lo_idx[tr], y[tr])
            te[va], cnt[va] = self.transform(lo_idx[va])
        return te, cnt