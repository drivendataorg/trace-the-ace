"""Model E2 feature matrix: the training-side wrapper around ``magnificat.model_e2``.

``magnificat.model_e2`` holds the E2 column spec (drops, twins, rates) and is shared with the
inference code (``models/submission/main.py`` applies it in its XGBoost branch). This module builds
the full Model E matrix from the training data and passes it to ``build_e2_matrix``, which adds
the E2-derived columns and selects the audited 70-column E2 subset. ``train_model_e3l.py``
imports ``build_e2_live``; run on its own, the script prints the objective-disjoint
cross-validation scores of Model E and E2.

Two ways to build the full Model E matrix:

  - ``build_e2_live()`` (default; the one the pipeline uses) computes every column from
    ``data/`` (transcripts and labels) with the same functions the inference code uses
    (``dfeat.build_dialogue_matrix``, ``features.build_design_matrix``,
    ``features.add_lo_curriculum_features``). Its only other input is ``nc_curriculum.csv``, a
    small public curriculum reference (shipped; ``scripts/build_nc_curriculum.py`` rebuilds it
    without competition data). Nothing is cached, so it cannot go stale relative to the feature
    code.
  - ``build_e2()`` (``--cached``) reads the cache files ``artifacts/dialogue_features.csv``,
    ``_features_train.pkl`` and ``model_e_supplement.csv``, which are not shipped.

Model E itself is not changed here.

Usage (from build/workspace/; not a reproduce.py step):
    python scripts/model_e2.py [--cached]
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import audit_lean_vs_full_e as A  # build_full_e, cv
from magnificat import data, features
from magnificat import dialogue_features as dfeat
from magnificat import model_e2 as e2spec


def build_full_e_live(train, meta, store=None):
    """Build the full Model E matrix from the transcripts, with no cache files.

    Uses the same feature code as the inference path (see the module docstring). The per-LO
    label statistics (``features.LO_DIFFICULTY_COLS``) are excluded.

    Args:
        train: training frame from ``data.load_train()`` (one row per response).
        meta: meta matrix from ``features.build_meta_matrix(train)``; unused, kept for the same
            signature as ``audit_lean_vs_full_e.build_full_e``.
        store: transcript store; defaults to ``data.default_store()``.

    Returns:
        ``(X, full_cols)``: the feature frame (dialogue, structured and curriculum columns) and
        the list of Model E column names in it.

    Raises:
        RuntimeError: if a structured column is missing, which means ``features.STRUCTURED_COLS``
            and ``features.transcript_stats()`` disagree in the current code.
    """
    store = store or data.default_store()
    dlg_X = dfeat.build_dialogue_matrix(train, store=store, show_progress=True)
    design_X = features.build_design_matrix(train, store=store, show_progress=True)

    t_cols = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
    available_t = [c for c in t_cols if c in design_X.columns]
    X = dlg_X.join(design_X[available_t])

    curriculum = features.load_lo_curriculum()  # optional; None is a graceful fallback
    nc_df = features.load_nc_curriculum_df()
    X = features.add_lo_curriculum_features(X, train, curriculum, nc_df=nc_df)

    full_cols = (list(dfeat.DIALOGUE_COLS) + list(dfeat.LO_COLS) + available_t
                 + list(features.LO_CURRICULUM_COLS))
    missing_t = [c for c in t_cols if c not in available_t]
    if missing_t:
        raise RuntimeError(
            f"build_full_e_live() is missing {len(missing_t)}/{len(t_cols)} "
            f"structured cols although it computes them from the transcripts, so "
            f"features.STRUCTURED_COLS and features.transcript_stats() disagree in "
            f"the current code (a bug in the code, not a stale cache): {missing_t}")
    return X, full_cols


def build_e2_live(train, meta, store=None):
    """Build the E2 matrix from the transcripts (no cache files).

    Args:
        train: training frame from ``data.load_train()``.
        meta: meta matrix from ``features.build_meta_matrix(train)`` (unused downstream).
        store: transcript store; defaults to ``data.default_store()``.

    Returns:
        ``(X, full_cols, e2_cols)``: the frame with the full E columns plus the E2-derived
        twins and rates, the full E column names, and the audited E2 column subset.
    """
    X_full, full_cols = build_full_e_live(train, meta, store=store)
    X, e2_cols = e2spec.build_e2_matrix(X_full, full_cols)
    return X, full_cols, e2_cols


def build_e2(train, meta):
    """Build the E2 matrix from cached feature files (the ``--cached`` path).

    Prefer ``build_e2_live``, which the pipeline uses.

    Args:
        train: training frame from ``data.load_train()``.
        meta: meta matrix from ``features.build_meta_matrix(train)``.

    Returns:
        ``(X, full_cols, e2_cols)`` as in ``build_e2_live``.
    """
    X_full, full_cols = A.build_full_e(train, meta)
    X, e2_cols = e2spec.build_e2_matrix(X_full, full_cols)
    return X, full_cols, e2_cols


def main():
    """Cross-validate Model E and E2 on the objective-disjoint folds and print both scores.

    Returns:
        None. Prints the column counts, log loss and AUC of E and E2, and their difference.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="use the cached-artifact path (artifacts/dialogue_features.csv "
                         "+ _features_train.pkl) instead of live computation. Only use this "
                         "if you already have and trust those caches.")
    args = ap.parse_args()

    train = data.load_train()
    y = train["is_correct"].to_numpy(float)
    lo = train["learning_objective_id"].to_numpy()
    meta = features.build_meta_matrix(train)
    if args.cached:
        X, full_cols, e2_cols = build_e2(train, meta)
        print(f"E  : {len(full_cols)} cols (cached path: full Model E incl. LOC and LOC-delta)")
    else:
        X, full_cols, e2_cols = build_e2_live(train, meta)
        print(f"E  : {len(full_cols)} cols (live path: E without LOC and LOC-delta, which "
              f"E2 drops anyway)")
    print(f"E2 : {len(e2_cols)} cols")
    ll_e, auc_e = A.cv(X, full_cols, y, lo)
    ll_2, auc_2 = A.cv(X, e2_cols, y, lo)
    print(f"\nE  objective-grouped CV: log loss {ll_e:.4f}  AUC {auc_e:.4f}")
    print(f"E2 objective-grouped CV: log loss {ll_2:.4f}  AUC {auc_2:.4f}")
    print(f"Δ E2−E: log loss {ll_2-ll_e:+.4f}  AUC {auc_2-auc_e:+.4f}  "
          f"(within ±0.0010 = no change)")


if __name__ == "__main__":
    main()
