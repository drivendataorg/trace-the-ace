#!/usr/bin/env python
"""Collapse the 44 per-code v2 rates onto pedagogical axes, at several widths.

Not a reproduce.py step. `magnificat.v2_features` mirrors the
collapse at inference (`build_v2_narrow_matrix`), and `check_v2_family_parity.py` imports this
module.

The collapse is exact: a family rate is a mean of soft probabilities over the same turn set,
and the mean of a sum is the sum of the means, so an axis column is the sum of its member
columns, whether the rate used a plain or an instructional denominator (the weighted means
share one denominator). Nothing is re-estimated, so this reads the family CSVs written by
`build_v2_families.py` instead of the per-turn tables.

The grouping is the pedagogical one (elicit, ask, tell, scaffold, positive feedback, negative
feedback, uptake, manage, admin).

In:  artifacts/v2fam_{rate,instr,trim}_{tag}.csv (--source, default rate)
Out: artifacts/v2fam_{narrow,narrows,narrowt,narrowstu}{_source}_{tag}.csv (the _source suffix
     only for --source instr or trim)

Usage (from build/workspace/):
  python scripts/build_v2_narrow.py --tag 122bC [--source rate|instr|trim]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Nine tutor axes over the 32 v2 codes. Every code appears exactly once: a code left out would
#: be silently discarded, and a code in two axes would be counted twice in the sums.
TUTOR_AXES = {
    "elicit": ["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP",
               "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
               "PROMPTING_SELF_CORRECTION", "RELEASE_HANDOFF"],
    "ask": ["ASKING_QUESTION", "TRANSFER_PROBE", "OFFERING_CHOICE"],
    "tell": ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
             "WORKED_DEMONSTRATION"],
    "scaffold": ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY",
                 "GIVING_EXAMPLE"],
    "fbpos": ["FEEDBACK_CORRECT", "GIVING_PRAISE"],
    "fbneg": ["FEEDBACK_INCORRECT"],
    "uptake": ["REVOICING", "RESTATING", "SUMMARIZING_PROGRESS", "FEEDBACK_NEUTRAL"],
    "manage": ["GUIDING_SESSION", "EXPLAINING_TOOL", "MOTIVATING_RELEVANCE", "TUTOR_OTHER"],
    "admin": ["TUTOR_SOCIAL", "TUTOR_TECHNICAL", "TUTOR_UNINTELLIGIBLE", "TUTOR_NO_MOVE"],
}

#: Six student axes. `SELF_CORRECTING` is grouped with `REASONING_SHOWN`.
STUDENT_AXES = {
    "bare": ["ANSWER_ONLY"],
    "reason": ["REASONING_SHOWN", "SELF_CORRECTING"],
    "insight": ["INSIGHT_DELIGHT"],
    "struggle": ["CONFUSION_EXPRESSED", "FRUSTRATION"],
    "seek": ["SEEKING_CLARIFICATION"],
    "sresid": ["STUDENT_SOCIAL", "STUDENT_TECHNICAL", "STUDENT_UNINTELLIGIBLE",
               "STUDENT_NO_MOVE", "STUDENT_OTHER"],
}


def collapse(df: pd.DataFrame, tpre: str, spre: str, out_pre: str,
             student_axes: bool) -> pd.DataFrame:
    """Sum member rates into axis rates, carrying the two derived ratios through unchanged.

    Keeps the index and ignores the key: the caller owns the key. This script writes
    session-keyed CSVs.

    Args:
        df: per-code family frame with ``{tpre}_<code>_rate`` columns, the two tutor ratios
            and ``{spre}_*`` student columns.
        tpre: tutor column prefix of the source family (e.g. "v2t").
        spre: student column prefix of the source family (e.g. "v2s").
        out_pre: prefix of the output columns (e.g. "v2n").
        student_axes: True to collapse the student codes onto STUDENT_AXES, False to copy
            them through per code.

    Returns:
        DataFrame with the same index: 9 tutor axis rates, the 2 ratios, and the student
        columns (6 axes or the per-code columns).
    """
    out = pd.DataFrame(index=df.index)
    for axis, codes in TUTOR_AXES.items():
        cols = [f"{tpre}_{c.lower()}_rate" for c in codes]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"missing tutor columns: {missing[:4]}")
        out[f"{out_pre}t_{axis}_rate"] = df[cols].sum(axis=1)
    # The two ratios are already axis-level quantities: 2 columns that carry the elicit-vs-tell
    # and feedback-valence constructs, which the sums cannot express as a level. Kept in every
    # width.
    for r in ("elicit_to_tell", "feedback_positive_ratio"):
        out[f"{out_pre}t_{r}"] = df[f"{tpre}_{r}"]
    if student_axes:
        for axis, codes in STUDENT_AXES.items():
            cols = [f"{spre}_{c.lower()}_rate" for c in codes]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise SystemExit(f"missing student columns: {missing[:4]}")
            out[f"{out_pre}s_{axis}_rate"] = df[cols].sum(axis=1)
    else:
        for c in [c for c in df.columns if c.startswith(f"{spre}_")]:
            out[c.replace(spre, f"{out_pre}s", 1)] = df[c]
    return out


def main() -> None:
    """Write the narrow blocks and check that the axes partition the codes exactly.

    Returns:
        None. Writes the four artifacts/v2fam_narrow*_{tag}.csv files; exits non-zero when
        the axis sums differ from the code sums by more than 1e-9.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="122bC")
    ap.add_argument("--source", default="rate", choices=["rate", "instr", "trim"],
                    help="which per-code family to collapse; the denominator carries through")
    args = ap.parse_args()

    pre = {"rate": ("v2t", "v2s"), "instr": ("v2ti", "v2si"), "trim": ("v2tt", "v2st")}
    tpre, spre = pre[args.source]
    src = ROOT / f"artifacts/v2fam_{args.source}_{args.tag}.csv"
    df = pd.read_csv(src)
    sfx = "" if args.source == "rate" else f"_{args.source}"

    for name, student_axes, out_pre in (("narrow", False, "v2n"),      # 9 + 2 + 12 = 23
                                        ("narrows", True, "v2m"),      # 9 + 2 +  6 = 17
                                        ("narrowt", None, "v2k"),      # 9 + 2      = 11
                                        ("narrowstu", None, "v2u")):   #          6 =  6
        out = collapse(df, tpre, spre, out_pre, True)
        if name == "narrowt":          # tutor half alone
            out = out[[c for c in out.columns if c.startswith(f"{out_pre}t_")]]
        elif name == "narrowstu":      # student half alone
            out = out[[c for c in out.columns if c.startswith(f"{out_pre}s_")]]
        elif not student_axes:
            out = collapse(df, tpre, spre, out_pre, False)
        out.insert(0, "session_id", df["session_id"].to_numpy())
        p = ROOT / f"artifacts/v2fam_{name}{sfx}_{args.tag}.csv"
        out.to_csv(p, index=False)
        print(f"wrote {p.name}  {len(out):,} sessions x {out.shape[1] - 1} cols "
              f"(the ntodst_ block has 21)", flush=True)

    # The collapse must be exact. A silent mismatch would make every narrow block a different
    # feature from the one described.
    chk = pd.read_csv(ROOT / f"artifacts/v2fam_narrow{sfx}_{args.tag}.csv")
    tot_axis = chk[[c for c in chk.columns if c.startswith("v2nt_")
                    and c.endswith("_rate")]].sum(axis=1)
    tot_code = df[[c for c in df.columns
                   if c.startswith(f"{tpre}_") and c.endswith("_rate")]].sum(axis=1)
    d = float((tot_axis - tot_code).abs().max())
    print(f"axis-sum identity: max |Σaxes − Σcodes| = {d:.3e}", flush=True)
    if d > 1e-9:
        raise SystemExit(f"FAILED: the 9 axes do not partition the 32 codes (max |Δ| {d:.3e})")


if __name__ == "__main__":
    main()
