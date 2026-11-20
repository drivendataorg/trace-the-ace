"""Model E2 feature spec, shared by the trainer and ``models/submission/main.py``.

Keeping one definition for both sides avoids a train/inference feature-matrix mismatch.

E2 = Model E's full feature set (125 numeric columns), minus:
  - compliance drops (4): the session_task features, which are computed across a
    session's rows and so risk disqualification under the competition rule against
    cross-sample features
  - source-artifact drops (9), noise drops (6), redundant raw-count
    drops (~8)
  - LOC + LOC-delta drops (19)
  - the extra and flat-timing drops (`DROP_EXTRA`, `DROP_FLAT_TIMING`)
plus:
  - the derived twin `student_turn_share` (turn balance, no existing equivalent)
  - Sandpiper's 15 raw counts as per-turn rates (every move kept)

The functions take any "full-E-shaped" matrix, built from cached artifacts at train time
(``src/training/scripts/model_e2.py``) or built live from transcripts at inference time
(the ``model_e2`` branch of ``models/submission/main.py``), and return the same 71-column
E2 subset either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- compliance: cross-sample features, not allowed ----------------------------
DROP_COMPLIANCE = ["session_n_responses", "session_n_distinct_los",
                   "session_lo_diversity", "session_lo_same"]
# --- noise drops --------------------------------------------------------------
DROP_NOISE = ["student_questions", "late_student_questions", "student_words_var",
              "student_max_turn_words", "total_chars", "n_unclear"]
# --- raw counts whose rate twin already exists (or is derived below) ----------
DROP_RAW_HAS_TWIN = ["n_utterances", "n_student", "n_tutor", "student_words",
                     "tutor_words", "n_question_marks", "student_short_turns",
                     "tutor_questions", "late_student_words", "late_tutor_questions"]
# --- source artifacts ---------------------------------------------------------
DROP_SOURCE = ["is_tsl", "n_background", "has_tutor_ph", "tutor_ph_count",
               "student_says_tutor_name", "tutor_says_tutor_name",
               "tutor_name_calls", "tutor_name_calls_per_turn", "has_name_call"]
# --- extra drops ---------------------------------------------------------------
# stu_ttr is a near-duplicate of student_ttr (both type-token ratio on student
# words); student_ttr (the standard whole-text TTR formula) is kept over stu_ttr
# (content words only, deduplicated per turn, then unioned).
DROP_EXTRA = ["tutor_praise", "stu_content_words", "longest_stu_utt_words", "pace_delta",
              "stu_ttr"]
# --- flat timing (timestamp-based) ---------------------------------------------
DROP_FLAT_TIMING = ["session_duration_secs", "avg_pause_secs", "avg_student_pause_secs"]

# --- Sandpiper 15 counts -> per-turn rates (every move kept) ------------------
SANDPIPER_COUNTS = ["tutor_feedback_incorrect", "tutor_backchannels",
    "tutor_prompts_explanation", "tutor_prompts_next_step",
    "tutor_prompts_self_correction", "tutor_prompts_alt_repr", "tutor_tag_questions",
    "tutor_gives_hint", "tutor_gives_example", "tutor_explains_procedural",
    "tutor_explains_conceptual", "tutor_gives_answer", "tutor_restating",
    "tutor_prior_knowledge_check", "student_explains"]
NEW_TWINS = ["student_turn_share"]  # turn balance; no existing equivalent


def _loc_drop_cols() -> list[str]:
    # Imported lazily to avoid import-order coupling. dialogue_features never
    # imports this module, so a module-level import would also work, but the lazy
    # import keeps this module importable on its own (e.g. from a bare script).
    from . import dialogue_features as dfeat
    return list(dfeat.LOC_COLS) + list(dfeat.LOC_DELTA_COLS)


def drop_cols() -> list[str]:
    """Return the full E2 drop list.

    Returns:
        Column names: compliance + noise + raw-with-twin + source + LOC + extra + timing.
    """
    return (DROP_COMPLIANCE + DROP_NOISE + DROP_RAW_HAS_TWIN + DROP_SOURCE
            + _loc_drop_cols() + DROP_EXTRA + DROP_FLAT_TIMING)


def add_e2_derived(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add `student_turn_share` and the Sandpiper per-turn rate columns.

    Any subset of SANDPIPER_COUNTS may be present (inference and training both have the
    full set; this stays defensive). Student moves are divided by `n_student`, tutor
    moves by `n_tutor`; a zero denominator gives 0.

    Args:
        X: feature table with `n_student`, `n_utterances`, `n_tutor` and the Sandpiper
            count columns. Not modified.

    Returns:
        ``(X_with_derived, sandpiper_rate_col_names)``: a copy of ``X`` with the new
        columns, and the names of the ``*_rate`` columns added.
    """
    X = X.copy()
    X["student_turn_share"] = (X["n_student"] / X["n_utterances"]).where(
        X["n_utterances"] > 0, 0.0)
    sand_rate_cols = []
    for c in SANDPIPER_COUNTS:
        if c not in X.columns:
            continue
        denom = X["n_student"] if c == "student_explains" else X["n_tutor"]
        X[c + "_rate"] = (X[c] / denom).where(denom > 0, 0.0)
        sand_rate_cols.append(c + "_rate")
    return X, sand_rate_cols


def e2_columns(full_cols: list[str], sand_rate_cols: list[str]) -> list[str]:
    """Select the E2 columns from the full Model E column list.

    Args:
        full_cols: the full Model E feature column names.
        sand_rate_cols: the Sandpiper rate columns from :func:`add_e2_derived`.

    Returns:
        The E2 column list: ``full_cols`` minus the drops and the raw Sandpiper counts,
        plus the rate columns and ``NEW_TWINS``.
    """
    drop = set(drop_cols() + SANDPIPER_COUNTS)
    return [c for c in full_cols if c not in drop] + sand_rate_cols + NEW_TWINS


def build_e2_matrix(X_full: pd.DataFrame, full_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build the E2 matrix; the single entry point for both the trainer and main.py.

    Args:
        X_full: a full-E-shaped feature table (built from cached artifacts at train time,
            or live at inference time).
        full_cols: its feature column names.

    Returns:
        ``(X_with_derived_cols, e2_cols)``.
    """
    X, sand_rate_cols = add_e2_derived(X_full)
    e2_cols = e2_columns(full_cols, sand_rate_cols)
    return X, e2_cols
