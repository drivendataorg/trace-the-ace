"""E2-robust: the generic ablation that builds the ablated training copies.

The shift to the Eedi rows (a provider whose transcripts lack some of the training source's
conventions) is simulated with generic random perturbations: features that exist only in the
training source are set to NaN, and the lexicon-rate and objective-overlap features are
attenuated by a random factor per ablated copy. The perturbation constants are not fitted on any
external data, and no externally licensed data is used.

``train_model_e3l.py`` imports ``ablate_full_row`` to build its ablated training copies. Run on
its own (not a reproduce.py step), the script trains a LightGBM per objective-disjoint fold with
and without ablated copies and prints the scores of both on the original and on ablated rows:
    python scripts/train_model_e2_robust.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import model_e2 as E2
from magnificat import baseline, data, features

# --- feature groups for the generic ablation (names as they appear in e2_cols) ---
NAN_OFF = ["n_long_pauses", "gap_cv", "wait_time_s",                 # timing (no timestamps off-TSL)
           "unclear_per_student_turn", "student_unclear_only"]       # [unclear] convention absent
LEXICON_RATES = ["tutor_elicit_rate", "tutor_tell_rate", "elicit_to_tell", "tutor_q_rate",
    "confirm_rate", "corrective_rate", "revoicing_rate", "uptake", "self_explanation_rate",
    "substantive_turn_rate", "confusion_rate", "resolution_rate", "hedge_rate",
    "ack_only_rate", "final_third_stu_q_rate", "help_deep_rate", "help_shallow_rate",
    "ends_resolved", "ends_confused", "recovered", "math_doing_rate"]
LO_OVERLAP = ["lo_cov_session", "lo_cov_student", "lo_cov_end", "lo_keyword_overlap",
    "lo_keyword_in_student", "lo_keyword_in_tutor", "lo_student_tutor_coverage_ratio",
    "lo_first_mention_frac", "lo_mention_density"]


def ablate_generic(X, cols, seed):
    """Simulate the Eedi regime with generic perturbations (no external constants).

    Sets the source-only columns (``NAN_OFF``) to NaN and multiplies, per row and column, the
    lexicon rates by U(0.2, 1.0) and the objective-overlap columns (``LO_OVERLAP``) by
    U(0.1, 0.6). Every column ending in ``_rate`` (the Sandpiper rates included) counts as a
    lexicon rate.

    Args:
        X: feature frame.
        cols: the columns to perturb; other columns are not returned.
        seed: seed of the numpy generator.

    Returns:
        A perturbed copy of ``X[cols]``.
    """
    rng = np.random.default_rng(seed)
    Z = X[cols].copy()
    lex = [c for c in cols if c in LEXICON_RATES or c.endswith("_rate")]
    lo = [c for c in cols if c in LO_OVERLAP]
    for c in cols:
        if c in NAN_OFF:
            Z[c] = np.nan
    if lex:
        Z[lex] = Z[lex].to_numpy() * rng.uniform(0.2, 1.0, size=(len(Z), len(lex)))
    if lo:
        Z[lo] = Z[lo].to_numpy() * rng.uniform(0.1, 0.6, size=(len(Z), len(lo)))
    return Z


def ablate_full_row(X, num, seed):
    """Return an ablated copy of a feature frame, perturbing only the numeric columns.

    The objective text and id (``learning_objective``, ``learning_objective_id``) are never
    perturbed: ablation-augmented training targets the transcript-derived features only.

    Args:
        X: feature frame, including the meta columns.
        num: the numeric transcript-derived columns to perturb (see ``ablate_generic``).
        seed: seed of the perturbation.

    Returns:
        A copy of ``X`` with ``X[num]`` replaced by ``ablate_generic(X, num, seed)``.
    """
    Z = X.copy()
    Zn = ablate_generic(X, num, seed=seed)
    Z[num] = Zn[num]
    return Z


def main():
    """Compare E2 trained without and with ablated copies on objective-disjoint folds.

    Returns:
        None. Prints, for each model, the full-view and ablated-view log loss, their balanced
        average and both AUCs.
    """
    train = data.load_train()
    y = train["is_correct"].to_numpy(float)
    lo = train["learning_objective_id"].to_numpy()
    meta = features.build_meta_matrix(train)
    X, _, e2_cols = E2.build_e2(train, meta)
    # numeric E2 cols only (the passthrough set of the LightGBM pipeline's dialogue branch);
    # learning_objective/_id are not in e2_cols: the pipeline reads them from X's own
    # columns (always present, see features.build_meta_matrix)
    num = [c for c in e2_cols if c not in ("learning_objective", "learning_objective_id")]
    print(f"E2-robust: {len(num)} numeric cols + meta (LO text/id)\n")

    def run(augmented):
        oof_full = np.zeros(len(y)); oof_abl = np.zeros(len(y))
        for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, lo)):
            Xf = X.iloc[tr]
            if augmented:
                Xa = ablate_full_row(X.iloc[tr], num, seed=100 + k)
                Xtr = pd.concat([Xf, Xa], ignore_index=True)
                ytr = np.concatenate([y[tr], y[tr]])
            else:
                Xtr, ytr = Xf, y[tr]
            pipe = baseline.build_meta_dialogue_gbm_pipeline(num)
            pipe.fit(Xtr, ytr)
            oof_full[te] = pipe.predict_proba(X.iloc[te])[:, 1]
            oof_abl[te] = pipe.predict_proba(ablate_full_row(X.iloc[te], num, seed=999))[:, 1]
        return oof_full, oof_abl

    print(f"{'model':<16}{'full LL':>9}{'ablated LL':>11}{'balanced':>10}{'full AUC':>9}{'abl AUC':>9}")
    for aug in (False, True):
        of, oa = run(aug)
        llf, lla = log_loss(y, of), log_loss(y, oa)
        bal = (llf + lla) / 2
        print(f"{'E2-augmented' if aug else 'E2-plain':<16}{llf:>9.4f}{lla:>11.4f}"
              f"{bal:>10.4f}{roc_auc_score(y,of):>9.4f}{roc_auc_score(y,oa):>9.4f}")


if __name__ == "__main__":
    main()
