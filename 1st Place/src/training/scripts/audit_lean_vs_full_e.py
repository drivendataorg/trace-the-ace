"""Diagnostic: full Model E against a lean feature subset.

Scores full Model E and the subset without the ~34 columns in DROP on the objective-grouped
5-fold out-of-fold view (log loss and AUROC). It uses the Model E feature caches
(artifacts/dialogue_features.csv, artifacts/_features_train.pkl,
artifacts/localized_features.csv, optionally artifacts/model_e_supplement.csv), which are not
shipped, and the Model E pipeline (`baseline.build_meta_dialogue_gbm_pipeline`).
`model_e2.py` imports `build_full_e` from here for its `--cached` path.

Not a reproduce.py step.
Usage (from build/workspace/, with the caches above):
    python scripts/audit_lean_vs_full_e.py            # full E vs lean
    python scripts/audit_lean_vs_full_e.py --decomp   # each audit group dropped on its own
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from magnificat import baseline, config, data, evaluate, features
from magnificat import dialogue_features as dfeat

DLG   = dfeat.DIALOGUE_COLS + dfeat.LO_COLS
T_COLS = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
CUR   = features.LO_CURRICULUM_COLS
LOC   = dfeat.LOC_COLS
DELTA = dfeat.LOC_DELTA_COLS

# ---- DROP: columns left out of the lean subset (only cols that exist in the E matrix) ----
DROP = [
    # source flags / artifacts
    "is_tsl", "n_background",
    "has_tutor_ph", "tutor_ph_count", "student_says_tutor_name", "tutor_says_tutor_name",
    "tutor_name_calls", "tutor_name_calls_per_turn", "has_name_call",
    "session_lo_same", "session_lo_diversity", "session_n_responses", "session_n_distinct_los",
    # unclear: keep the ratio, drop the whole-transcript count
    "n_unclear",
    # session timing (keep n_long_pauses, gap_cv, wait_time_s)
    "session_duration_secs", "avg_pause_secs", "avg_student_pause_secs",
    # student question counts
    "student_questions", "late_student_questions",
    # raw length measures (rate twins kept)
    "tutor_words", "total_chars", "student_max_turn_words",
    "n_utterances", "n_student", "n_tutor", "student_words", "n_question_marks",
    "student_short_turns", "late_student_words",
    # word-count variance
    "student_words_var",
    # LOC + LOC-delta
    *LOC, *DELTA,
    # LO word-overlap
    *DLG,  # removed again below: only the LO_COLS subset and LO_TRANSCRIPT_COLS are dropped
]
# refine: from DLG we only drop the LO-overlap coverage cols, keep dialogue mechanics
DROP = [c for c in DROP if c not in DLG]
DROP += list(dfeat.LO_COLS)                      # lo_cov_session/student/end
DROP += list(features.LO_TRANSCRIPT_COLS)        # lo_keyword_*, lo_mention_density, ...


def _load_supplement(train, meta):
    """Load Model E's Sandpiper supplement columns, aligned to the response rows.

    The three columns (tutor_explains_conceptual, tutor_prior_knowledge_check,
    error_response_quality) are cached separately from _features_train.pkl and keyed by
    session_id, not response_id. Without this join, a full-E rebuild silently lacks the three
    columns.

    Args:
        train: training frame with response_id and session_id.
        meta: meta matrix indexed by response_id.

    Returns:
        DataFrame indexed like ``meta`` with the supplement columns, or with no columns when
        artifacts/model_e_supplement.csv does not exist (e.g. runs on synthetic data).
    """
    path = config.ARTIFACTS_DIR / "model_e_supplement.csv"
    if not path.exists():
        return pd.DataFrame(index=meta.index)
    supp = pd.read_csv(path)
    sess_to_resp = train[["response_id", "session_id"]].set_index("response_id")
    merged = sess_to_resp.join(supp.set_index("session_id"), on="session_id")
    return merged.drop(columns="session_id").reindex(meta.index)


def _check_full_e_completeness(X_e, dlg, t_X_full, supp, curriculum, nc_df,
                                available_t, available_loc, available_delta):
    """Raise with a diagnosis when the assembled Model E matrix lacks expected columns.

    The column filters in `build_full_e` (`[c for c in COLS if c in df.columns]`) are
    permissive so that a partial build (a stale or partial artifacts/ cache) does not crash,
    which also means nothing else reports that the build was partial; this check does.

    Args:
        X_e: the assembled feature frame.
        dlg: dialogue-feature frame (unused, kept for the diagnosis signature).
        t_X_full: the full cached structured-feature frame from _features_train.pkl.
        supp: supplement frame from `_load_supplement`.
        curriculum: LO curriculum lookup, or None when lo_curriculum.json is missing.
        nc_df: NC curriculum frame, or None when nc_curriculum.csv is missing.
        available_t: structured/Sandpiper columns found.
        available_loc: LOC columns found (unused).
        available_delta: LOC-delta columns found (unused).

    Raises:
        RuntimeError: when structured/Sandpiper or LO-curriculum columns are missing.
    """
    missing_t = [c for c in T_COLS if c not in available_t]
    missing_cur = [c for c in CUR if c not in X_e.columns]
    problems = []
    if missing_t:
        problems.append(
            f"{len(missing_t)}/{len(T_COLS)} structured/Sandpiper cols missing from "
            f"_features_train.pkl + model_e_supplement.csv: {missing_t}\n"
            f"    -> _features_train.pkl has {t_X_full.shape[1]} cols total, "
            f"model_e_supplement.csv contributed {list(supp.columns)}.\n"
            f"    -> likely cause: this artifacts/_features_train.pkl is a different or "
            f"older build than the one Model E was built from. Check that you point at "
            f"the same artifacts/ directory (same file; compare size and mtime) and do "
            f"not proceed with a mismatched cache.")
    if missing_cur:
        problems.append(
            f"{len(missing_cur)}/{len(CUR)} LO-curriculum cols missing: {missing_cur}\n"
            f"    -> curriculum lookup {'FOUND' if curriculum else 'MISSING (lo_curriculum.json)'}, "
            f"NC dataframe {'FOUND' if nc_df is not None else 'MISSING (nc_curriculum.csv)'}.")
    if problems:
        raise RuntimeError(
            "build_full_e() produced an incomplete feature matrix; refusing to "
            "continue, because the run would otherwise train and score a weaker "
            "model without any visible error. Problems found:\n\n"
            + "\n\n".join(problems)
            + "\n\nFix the missing artifacts/ files (or point at the correct "
              "artifacts/ directory) before re-running."
        )


def build_full_e(train, meta):
    """Assemble the full Model E feature matrix from the cached feature files.

    Args:
        train: training frame from ``data.load_train()``.
        meta: meta matrix from ``features.build_meta_matrix(train)``, indexed by response_id.

    Returns:
        ``(X_e, all_num)``: the joined feature frame and the list of numeric Model E columns
        (dialogue and LO overlap, structured, curriculum, LOC and LOC-delta).

    Raises:
        RuntimeError: from `_check_full_e_completeness` when columns are missing.
    """
    dlg = pd.read_csv(config.ARTIFACTS_DIR / "dialogue_features.csv", index_col=0)
    t_X_full, _ = joblib.load(config.ARTIFACTS_DIR / "_features_train.pkl")
    available_t = [c for c in T_COLS if c in t_X_full.columns]
    t = t_X_full[available_t].reindex(meta.index)
    supp = _load_supplement(train, meta)
    supp = supp[[c for c in supp.columns if c not in t.columns]]  # some caches already have these baked in
    t = t.join(supp)
    available_t = available_t + list(supp.columns)
    X_base = meta.join(dlg.reindex(meta.index)[DLG])
    X_c = X_base.join(t)
    curriculum = features.load_lo_curriculum()
    nc_df = features.load_nc_curriculum_df()
    X_d = features.add_lo_curriculum_features(X_c, train, curriculum, nc_df=nc_df)
    loc_df = pd.read_csv(config.ARTIFACTS_DIR / "localized_features.csv", index_col=0).reindex(meta.index)
    available_loc = [c for c in LOC if c in loc_df.columns]
    X_e = X_d.join(loc_df[available_loc])
    X_e = dfeat.add_loc_delta_features(X_e)
    available_delta = [c for c in DELTA if c in X_e.columns]
    _check_full_e_completeness(X_e, dlg, t_X_full, supp, curriculum, nc_df,
                                available_t, available_loc, available_delta)
    all_num = DLG + available_t + CUR + available_loc + available_delta
    return X_e, all_num


def cv(X, cols, y, lo):
    """Score a column set by objective-grouped 5-fold out-of-fold prediction.

    Args:
        X: feature frame.
        cols: numeric columns to use.
        y: binary labels, shape (n,).
        lo: learning_objective_id per row (the CV groups).

    Returns:
        ``(log_loss, auroc)`` of the out-of-fold predictions.
    """
    oof = evaluate.group_oof(
        baseline.build_meta_dialogue_gbm_pipeline(cols), X, y, lo, n_splits=5, n_jobs=1)
    return log_loss(y, oof), roc_auc_score(y, oof)


def main():
    """Cross-validate full Model E and the lean subset and print both and their difference.

    Returns:
        None (prints only).
    """
    train = data.load_train()
    y = train["is_correct"].to_numpy(float)
    lo = train["learning_objective_id"].to_numpy()
    meta = features.build_meta_matrix(train)
    X_e, full_cols = build_full_e(train, meta)

    lean_cols = [c for c in full_cols if c not in set(DROP)]
    print(f"full E : {len(full_cols)} numeric cols")
    print(f"lean   : {len(lean_cols)} numeric cols  ({len(full_cols) - len(lean_cols)} dropped)")
    dropped = sorted(set(full_cols) - set(lean_cols))
    print(f"dropped: {dropped}\n")

    ll_f, auc_f = cv(X_e, full_cols, y, lo)
    print(f"FULL E : log loss {ll_f:.4f}  AUC {auc_f:.4f}")
    ll_l, auc_l = cv(X_e, lean_cols, y, lo)
    print(f"LEAN   : log loss {ll_l:.4f}  AUC {auc_l:.4f}")
    print(f"\nΔ lean − full: log loss {ll_l - ll_f:+.4f}  AUC {auc_l - auc_f:+.4f}")
    print("(+log loss = lean worse; we treated differences within ±0.0010 as no change)")


if __name__ == "__main__" and "--decomp" not in sys.argv:
    main()


def group_decomp():
    """Drop each audit group on its own and print the change in log loss against full E.

    Returns:
        None (prints one row per group).
    """
    train = data.load_train()
    y = train["is_correct"].to_numpy(float)
    lo = train["learning_objective_id"].to_numpy()
    meta = features.build_meta_matrix(train)
    X_e, full_cols = build_full_e(train, meta)
    fullset = set(full_cols)
    groups = {
        "source_artifacts": [c for c in ["is_tsl","n_background","has_tutor_ph","tutor_ph_count",
            "student_says_tutor_name","tutor_says_tutor_name","tutor_name_calls",
            "tutor_name_calls_per_turn","has_name_call"] if c in fullset],
        "session_task": [c for c in ["session_lo_same","session_lo_diversity",
            "session_n_responses","session_n_distinct_los"] if c in fullset],
        "unclear_count": [c for c in ["n_unclear"] if c in fullset],
        "flat_timing": [c for c in ["session_duration_secs","avg_pause_secs","avg_student_pause_secs"] if c in fullset],
        "nosignal_qcounts": [c for c in ["student_questions","late_student_questions"] if c in fullset],
        "redundant_counts": [c for c in ["tutor_words","total_chars","student_max_turn_words",
            "n_utterances","n_student","n_tutor","student_words","n_question_marks",
            "student_short_turns","late_student_words","student_words_var"] if c in fullset],
        "LOC_delta": [c for c in (LOC + DELTA) if c in fullset],
        "LO_overlap": [c for c in (list(dfeat.LO_COLS) + list(features.LO_TRANSCRIPT_COLS)) if c in fullset],
    }
    ll_f, _ = cv(X_e, full_cols, y, lo)
    print(f"FULL E baseline: {ll_f:.4f}\n")
    print(f"{'group':<20} {'ncols':>5} {'drop-only ΔLL':>14}")
    for name, cols in groups.items():
        keep = [c for c in full_cols if c not in set(cols)]
        ll, _ = cv(X_e, keep, y, lo)
        print(f"{name:<20} {len(cols):>5} {ll - ll_f:>+14.4f}")


if __name__ == "__main__" and "--decomp" in sys.argv:
    group_decomp()
