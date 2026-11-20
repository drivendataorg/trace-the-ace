#!/usr/bin/env python
"""Compile the interview-codebook labels into session-level feature blocks.

The codebook is described in `src/training/docs/29_interviews_and_labels.md`. Two blocks with
the same column names:

  --mode oracle     from the raw LLM records; covers only annotated sessions, NaN elsewhere
                    (prefix d29_)
  --mode distilled  from the distilled heads; covers all 22,821 sessions (out-of-fold
                    probabilities where the annotator ran, the all-data fit elsewhere)
                    (prefix d29dst_; the block the shipped XGBoost model uses)

Session grain: one row per session, shared by every objective of that session.

Denominators: confidence is only defined on turns where the student asserts something, so its
denominator is answering turns, not all student turns; otherwise the feature would measure
how talkative the student is.

In:  artifacts/doc29_records_14b/doc29_raw_records_14b.parquet   (oracle)
     artifacts/doc29_turn_probs.pkl.gz + artifacts/doc29_clf.joblib  (distilled)
     data/train_features.csv, data/train_transcripts/                (distilled)
Out: artifacts/doc29_features_oracle.csv | artifacts/doc29_features_distilled.csv

Usage (reproduce.py step 04a, from build/workspace/):
  OMP_NUM_THREADS=1 python scripts/build_doc29_features.py --mode distilled
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from magnificat import data  # noqa: E402

from distil_doc29_labels import HEADS, build_text, head_name  # noqa: E402

TUTOR_CODES = [v for p, _f, v in HEADS if p == "doc29_tutor"]
LATE = 2.0 / 3.0          # "final third" boundary on position_frac


def _agg(g: pd.DataFrame, cols: dict[str, pd.Series], denom: pd.Series,
         prefix: str) -> dict:
    """Rate of each soft or hard indicator over a denominator, overall and in the final third.

    Args:
        g: one session's rows, with a `position_frac` column (0 = first turn, 1 = last).
        cols: feature name -> per-row indicator (0/1) or probability, aligned with ``g``.
        denom: per-row weight defining the denominator (1 for every turn, or P(answering)).
        prefix: column-name prefix (d29_ or d29dst_).

    Returns:
        Dict with ``{prefix}{name}_rate`` and ``{prefix}{name}_late`` per feature; NaN where
        the denominator is zero.
    """
    out = {}
    d = float(denom.sum())
    late = g["position_frac"].fillna(0.0) >= LATE
    dl = float((denom * late).sum())
    for name, v in cols.items():
        out[f"{prefix}{name}_rate"] = float((v * denom).sum() / d) if d > 0 else np.nan
        out[f"{prefix}{name}_late"] = float((v * denom * late).sum() / dl) if dl > 0 else np.nan
    return out


def compile_block(tut: pd.DataFrame, stu: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate per-utterance tutor and student labels into one row per session.

    Args:
        tut: tutor utterances, one ``tut_<code>`` column per tutor move holding a 0/1
            indicator (oracle) or a probability (distilled), plus session_id and
            position_frac.
        stu: student utterances with the ``stu_*`` head columns in the same form.
        prefix: column-name prefix (d29_ or d29dst_).

    Returns:
        DataFrame with one row per session: session_id and the rate, late-share, coverage,
        answering, reasoning-share, confidence-contrast and delight columns.
    """
    rows = {}

    for sid, g in tut.groupby("session_id", sort=False):
        one = pd.Series(1.0, index=g.index)
        cols = {c: g[f"tut_{c.lower()}"] for c in TUTOR_CODES}
        r = _agg(g, cols, one, prefix)
        # Coverage: the share of tutor turns that performed any codebook move, kept as an
        # explicit feature rather than an implicit denominator.
        r[f"{prefix}any_move_rate"] = float(
            np.clip(sum(g[f"tut_{c.lower()}"] for c in TUTOR_CODES), 0, 1).mean())
        # No raw turn count here: it is length-coupled and the handcrafted block already
        # carries turn volume, so it would be a fragile duplicate.
        rows.setdefault(sid, {}).update(r)

    for sid, g in stu.groupby("session_id", sort=False):
        one = pd.Series(1.0, index=g.index)
        ans_only, reas = g["stu_answer_only"], g["stu_reasoning_shown"]
        answering = (ans_only + reas).clip(0, 1)          # complement = NOT_ANSWERING
        hed, unm, ass = g["stu_hedged"], g["stu_unmarked"], g["stu_assertive"]
        del_, fru = g["stu_insight_delight"], g["stu_frustration"]

        r = _agg(g, {"answer_only": ans_only, "reasoning_shown": reas,
                     "insight_delight": del_, "frustration": fru}, one, prefix)
        # confidence is defined on answering turns only (see the module docstring)
        r.update(_agg(g, {"hedged": hed, "unmarked": unm, "assertive": ass},
                      answering, prefix))
        r[f"{prefix}answering_rate"] = float(answering.mean())
        # §5.3: of the turns where the student answers, the share that shows work
        tot = float((ans_only + reas).sum())
        r[f"{prefix}reasoning_share"] = float(reas.sum() / tot) if tot > 0 else np.nan
        # §7.3: net epistemic stance, bounded to [-1, 1] and length-free by construction
        hs = float(hed.sum())
        as_ = float(ass.sum())
        r[f"{prefix}confidence_contrast"] = ((as_ - hs) / (as_ + hs)
                                             if (as_ + hs) > 0 else np.nan)
        # §7.4: "an aha in the last third is not the same as one in the first"
        w = float(del_.sum())
        r[f"{prefix}delight_pos_mean"] = (
            float((del_ * g["position_frac"].fillna(0.5)).sum() / w) if w > 0 else np.nan)
        r[f"{prefix}delight_any"] = float(min(w, 1.0))
        rows.setdefault(sid, {}).update(r)

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "session_id"
    return out.reset_index()


def oracle_frame(records: pd.DataFrame):
    """Turn the raw LLM records into per-utterance 0/1 indicator frames.

    Args:
        records: raw records from `annotate_doc29_labels.py` (annotation_type, code and
            the student fields).

    Returns:
        ``(tut, stu)`` in the form `compile_block` expects.
    """
    tut = records[records.annotation_type == "doc29_tutor"].copy()
    stu = records[records.annotation_type == "doc29_student"].copy()
    for c in TUTOR_CODES:
        tut[f"tut_{c.lower()}"] = (tut["code"] == c).astype(float)
    for _p, f, v in HEADS:
        if _p == "doc29_student":
            stu[head_name(_p, f, v)] = (stu[f] == v).astype(float)
    return tut, stu


def distilled_frame(probs: pd.DataFrame, clf, transcripts: Path,
                    all_sessions: list[str]):
    """Per-utterance head probabilities: out-of-fold where the annotator ran, all-data fit elsewhere.

    An out-of-fold score is the unbiased one for a session the heads were trained on, and the
    all-data fit is the best available estimate for a session no fold saw. Neither path sees
    `is_correct`.

    Args:
        probs: out-of-fold per-utterance probabilities from `distil_doc29_labels.py`.
        clf: the all-data fit (dict pass name -> {"vec", "heads"}).
        transcripts: directory of the training transcripts.
        all_sessions: every training session id.

    Returns:
        ``(tut, stu)`` in the form `compile_block` expects; a head missing from the fit gets
        a column of zeros.
    """
    seen = set(probs.session_id.unique())
    todo = [s for s in all_sessions if s not in seen]
    print(f"distilled: {len(seen):,} sessions from OOF, {len(todo):,} to score with "
          f"the all-data fit", flush=True)

    frames = {"doc29_tutor": [probs[probs.annotation_type == "doc29_tutor"]],
              "doc29_student": [probs[probs.annotation_type == "doc29_student"]]}

    store = data.TranscriptStore(transcripts)
    BATCH = 400
    for pass_name, role, marker in (("doc29_tutor", "tutor", "T"),
                                    ("doc29_student", "student", "S")):
        heads = clf[pass_name]["heads"]
        vec = clf[pass_name]["vec"]
        t0 = time.time()
        for bi in range(0, len(todo), BATCH):
            batch = todo[bi:bi + BATCH]
            rec = []
            for sid in batch:
                try:
                    df = store.load(sid)
                except Exception:
                    continue
                if df.empty:
                    continue
                sub = df[df["role"].astype(str) == role]
                n = max(len(df) - 1, 1)
                for i, uid in enumerate(sub["utterance_id"].tolist()):
                    rec.append({"session_id": sid, "utterance_id": uid,
                                "position_frac": i / max(len(sub) - 1, 1)})
            if not rec:
                continue
            rdf = pd.DataFrame(rec)
            rdf["annotation_type"] = pass_name
            txt = build_text(rdf, transcripts, role, marker)
            if txt.empty:
                continue
            X = vec.transform(txt["text"].values)
            for h, m in heads.items():
                txt[h] = m.predict_proba(X)[:, 1].astype(np.float32)
            frames[pass_name].append(
                txt[["session_id", "utterance_id", "position_frac", "annotation_type"]
                    + list(heads)])
            if bi % (BATCH * 5) == 0:
                el = time.time() - t0
                done = min(bi + BATCH, len(todo))
                print(f"  [{pass_name}] {done:,}/{len(todo):,} sessions, "
                      f"{el/60:.1f} min elapsed", flush=True)

    tut = pd.concat(frames["doc29_tutor"], ignore_index=True)
    stu = pd.concat(frames["doc29_student"], ignore_index=True)
    for c in TUTOR_CODES:                       # heads that fell back get a 0 column
        if f"tut_{c.lower()}" not in tut.columns:
            tut[f"tut_{c.lower()}"] = 0.0
    for _p, f, v in HEADS:
        if _p == "doc29_student" and head_name(_p, f, v) not in stu.columns:
            stu[head_name(_p, f, v)] = 0.0
    return tut.fillna({f"tut_{c.lower()}": 0.0 for c in TUTOR_CODES}), stu


def main() -> None:
    """Build the oracle or the distilled block and write it.

    Returns:
        None. Writes artifacts/doc29_features_{oracle,distilled}.csv (or ``--out``) and prints
        a summary of the columns.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "distilled"], required=True)
    ap.add_argument("--records", type=Path,
                    default=ROOT / "artifacts/doc29_records_14b/doc29_raw_records_14b.parquet")
    ap.add_argument("--probs", type=Path, default=ROOT / "artifacts/doc29_turn_probs.pkl.gz")
    ap.add_argument("--clf", type=Path, default=ROOT / "artifacts/doc29_clf.joblib")
    # In the workspace the training transcripts are under data/.
    ap.add_argument("--transcripts", type=Path, default=ROOT / "data/train_transcripts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    prefix = "d29_" if args.mode == "oracle" else "d29dst_"
    out = args.out or ROOT / f"artifacts/doc29_features_{args.mode}.csv"

    if args.mode == "oracle":
        recs = pd.read_parquet(args.records)
        tut, stu = oracle_frame(recs)
    else:
        import joblib  # noqa: PLC0415
        probs = pd.read_pickle(args.probs, compression="gzip")
        clf = joblib.load(args.clf)
        train = data.load_train()
        tut, stu = distilled_frame(probs, clf, args.transcripts,
                                   list(pd.unique(train["session_id"])))

    block = compile_block(tut, stu, prefix)
    block.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(block):,} sessions x {block.shape[1]-1} features)")
    num = block.drop(columns=["session_id"])
    print(f"  non-null: {num.notna().mean().mean():.1%}")
    print(num.describe().T[["mean", "std"]].round(4).to_string())


if __name__ == "__main__":
    main()
