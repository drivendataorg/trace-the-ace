#!/usr/bin/env python
"""Session-level NTO tutor-move rates from the LLM codes (oracle rates).

E2 detects tutor moves with regexes (`tutor_gives_hint_rate`, `confirm_rate`,
`corrective_rate`, ...).

This block counts the move families E2 already counts, from the LLM annotator's codes instead
of regexes. The codes exist only for training sessions, so the table is a diagnostic, not a
model input. Step 03c (`distil_nto_moves.py`) reads it for its report; that script imports
FAMILIES and SHARDS from here, and `distil_v2_moves.py` imports FAMILIES.

In:  external/dialogue-kt/results/nto_moves_shard*.csv
Out: artifacts/nto_rate_features.csv  (one row per session_id)

Usage (reproduce.py step 03b, from build/workspace/):
  python scripts/build_nto_rate_features.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHARDS = sorted((ROOT / "external/dialogue-kt/results").glob("nto_moves_shard*.csv"))

# NTO Table 1 families: the diagnostic codes (reports about the student) are kept separate
# from the instructional choices.
FAMILIES = {
    "fb_correct": ["FEEDBACK_CORRECT"],
    "fb_incorrect": ["FEEDBACK_INCORRECT"],
    "fb_neutral": ["FEEDBACK_NEUTRAL"],
    "praise": ["PRAISING_PROCESS", "PRAISING_OUTCOME", "PRAISING_TRAITS"],
    "encourage": ["ENCOURAGING"],
    "rapport": ["BUILDING_RAPPORT"],
    "prompt_self_expl": ["PROMPTING_SELF_EXPLANATION"],
    "prompt_next_step": ["PROMPTING_NEXT_STEP"],
    "prompt_self_correct": ["PROMPTING_SELF_CORRECTION"],
    "probe_understand": ["PROBING_UNDERSTAND", "PROBING_UNDERSTANDING"],
    "probe_prior": ["PROBING_PRIOR_KNOWLEDGE"],
    "revoicing": ["REVOICING"],
    "clarify": ["ASKING_TO_CLARIFY_CONTEXT"],
    "hint": ["GIVING_HINT"],
    "strategize": ["STRATEGIZING"],
    "explain_conceptual": ["EXPLAINING_CONCEPTUAL"],
    "explain_procedural": ["EXPLAINING_PROCEDURAL"],
    "give_answer": ["GIVING_ANSWER"],
    "technical": ["TECHNICAL_ISSUES"],
}


def main() -> None:
    """Compute per-session NTO move-family rates from the annotation shards.

    Returns:
        None. Writes artifacts/nto_rate_features.csv: per session, `nto_<family>_rate`
        (share of coded tutor turns with at least one code of the family) plus
        `nto_elicit_to_tell` and `nto_feedback_positive_ratio`.
    """
    df = pd.concat([pd.read_csv(p) for p in SHARDS], ignore_index=True)
    df = df.drop_duplicates(["session_id", "turn"])
    df["codes"] = df["codes"].fillna("")
    print(f"NTO codes: {len(df):,} turns / {df.session_id.nunique():,} sessions")

    code_of = {c: fam for fam, cs in FAMILIES.items() for c in cs}
    # Multi-label: a turn can carry several codes, so explode before counting.
    ex = df.assign(code=df.codes.str.split("|")).explode("code")
    ex["fam"] = ex.code.str.strip().map(code_of)
    seen = set(ex.code.dropna().str.strip()) - set(code_of)
    if seen:
        print(f"  WARNING: {len(seen)} codes not mapped to a family (ignored): "
              f"{sorted(seen)[:8]}{' …' if len(seen) > 8 else ''}")

    n_turns = df.groupby("session_id").size().rename("n_coded_turns")
    wide = (ex.dropna(subset=["fam"])
              .drop_duplicates(["session_id", "turn", "fam"])   # one hit per turn per family
              .groupby(["session_id", "fam"]).size().unstack(fill_value=0))
    wide = wide.reindex(columns=sorted(FAMILIES), fill_value=0)
    rates = wide.div(n_turns.reindex(wide.index), axis=0)
    rates.columns = [f"nto_{c}_rate" for c in rates.columns]

    # the two ratios E2 already builds from regex, rebuilt on the LLM coding
    rates["nto_elicit_to_tell"] = (
        (rates.nto_prompt_self_expl_rate + rates.nto_prompt_next_step_rate
         + rates.nto_probe_understand_rate + rates.nto_probe_prior_rate)
        / (rates.nto_explain_conceptual_rate + rates.nto_explain_procedural_rate
           + rates.nto_give_answer_rate + 1e-3))
    rates["nto_feedback_positive_ratio"] = (
        rates.nto_fb_correct_rate
        / (rates.nto_fb_correct_rate + rates.nto_fb_incorrect_rate + 1e-3))

    out = rates.reset_index()
    dst = ROOT / "artifacts/nto_rate_features.csv"
    out.to_csv(dst, index=False)
    print(f"wrote {dst}  ({len(out):,} sessions x {out.shape[1] - 1} features)")
    print(rates.mean().round(4).to_string())


if __name__ == "__main__":
    main()
