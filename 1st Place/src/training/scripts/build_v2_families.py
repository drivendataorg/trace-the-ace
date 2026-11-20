#!/usr/bin/env python
"""Build the aggregations ("families") of the v2 per-turn move probabilities into XGBoost blocks.

`distil_v2_moves.py` writes ``artifacts/{prefix}_turn_probs_{tag}.pkl.gz``, one soft
probability per (session, turn, code) for 32 tutor codes and 12 student codes over the whole
corpus. This script aggregates those probabilities in several ways, one file per family. The
shipped XGBoost model uses the `shrunk` family (`v2fam_shrunk_122bC.csv`, 46 v2ht_/v2hs_
columns).

`shrunk` (definition in :mod:`magnificat.v2_agg`): each code's mean probability over the turns
of that role, shrunk towards the corpus mean (kappa 25) so that its noise does not depend on
transcript length, plus the tutor ratios `elicit_to_tell` and `feedback_positive_ratio`. It
is a session-level scalar, constant across the objectives of a session.

The other families (`rate`, `shrunkeb`, `shrunkk35`, `shrunkq`, `win12`, `win24`, `win48`,
`instr`, `trim`, `bound`, `slope`, `disp`, `trans`, `ratio`, `qual`, `lowin`) are written to
their own files too.

In:  artifacts/v2t_turn_probs_<tag>.pkl.gz, artifacts/v2s_turn_probs_<tag>.pkl.gz (--tutor,
     --student), data/train_features.csv, data/train_transcripts/ (for `lowin`)
Out: artifacts/v2fam_{family}_{tag}.csv  (session-keyed, except `lowin`, which is
     response-keyed)

Usage (reproduce.py step 04b, from build/workspace/). --tutor and --student are required for
the shipped block: the defaults point at the un-suffixed _122b tables, which build a different
block.
  OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/build_v2_families.py --tag 122bC --jobs 8 \
    --tutor artifacts/v2t_turn_probs_v2ext_122bC.pkl.gz \
    --student artifacts/v2s_turn_probs_v2stu_122bC.pkl.gz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magnificat import data, v2_agg  # noqa: E402
from magnificat.evidence_features import build_turns  # noqa: E402
from magnificat.histpos_features import _lo_words  # noqa: E402

#: +/-N paired turns around every turn whose text hits the objective's keywords (a paired turn
#: is ~2 utterances).
LO_PAD_TURNS = 2

#: Axes for `slope` and `disp`: named subsets of the tutor and the student codes.
AXES_T = {
    "ask": v2_agg.ASK,
    "tell": v2_agg.TELL,
    "fb_pos": ["FEEDBACK_CORRECT", "GIVING_PRAISE"],
    "fb_neg": ["FEEDBACK_INCORRECT"],
    "hint": ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY"],
    "admin": v2_agg.TUTOR_RESID,
}
AXES_S = {
    "reason": v2_agg.STU_ENGAGE,
    "struggle": v2_agg.STU_STRUGGLE,
    "bare": ["ANSWER_ONLY"],
}


def _session_arrays(tg: pd.DataFrame, sg: pd.DataFrame, tcols, scols):
    """Align the tutor and student per-turn tables of one session onto one turn axis.

    They are different subsets of `turn_text.pkl.gz`: the student pass scores every paired
    turn (the table already requires student text), the tutor pass only turns with a tutor
    utterance. Rates must be taken over each pass's own population (as in the shipped block
    and the classifier's training), but transitions and the LO window need the two side by
    side. So both are returned: masks for rates, and zero-filled aligned matrices for anything
    that reads across roles. A zero row on the tutor side means "no tutor utterance in this
    turn", not missing data.

    Args:
        tg: the session's tutor rows (``pos`` plus one probability column per code).
        sg: the session's student rows, in the same form.
        tcols: tutor code columns, in order.
        scols: student code columns, in order.

    Returns:
        ``(T, S, tmask, smask, pos)``: float32 matrices of shape (n_turns, len(tcols)) and
        (n_turns, len(scols)), boolean masks of the turns each pass scored, and the sorted
        union of turn positions that defines the axis.
    """
    pos = np.union1d(tg["pos"].to_numpy(), sg["pos"].to_numpy())
    where = {p: i for i, p in enumerate(pos)}
    n = len(pos)
    T = np.zeros((n, len(tcols)), dtype=np.float32)
    S = np.zeros((n, len(scols)), dtype=np.float32)
    tmask = np.zeros(n, dtype=bool)
    smask = np.zeros(n, dtype=bool)
    ti = np.fromiter((where[p] for p in tg["pos"].to_numpy()), int, len(tg))
    si = np.fromiter((where[p] for p in sg["pos"].to_numpy()), int, len(sg))
    T[ti] = tg[tcols].to_numpy(np.float32)
    S[si] = sg[scols].to_numpy(np.float32)
    tmask[ti] = True
    smask[si] = True
    return T, S, tmask, smask, pos


def session_families(T, S, tmask, smask, tcols, scols) -> dict:
    """Compute every session-keyed family for one session.

    Args:
        T, S, tmask, smask: aligned matrices and masks from `_session_arrays`.
        tcols: tutor code columns.
        scols: student code columns.

    Returns:
        Flat dict ``"<family>|<column>" -> value``; `main` splits it into one file per family.
    """
    out = {}
    wT = v2_agg.instructional_weight(T, tcols, v2_agg.TUTOR_RESID)
    wS = v2_agg.instructional_weight(S, scols, v2_agg.STUDENT_RESID)

    def emit(fam, pre, vals, cols):
        for c, v in zip(cols, vals):
            out[f"{fam}|{pre}_{c.lower()}_rate"] = float(v)

    def ratios(fam, pre, vals, cols):
        ix = {c: j for j, c in enumerate(cols)}
        el = sum(vals[ix[c]] for c in v2_agg.ELICIT if c in ix)
        te = sum(vals[ix[c]] for c in v2_agg.TELL if c in ix)
        out[f"{fam}|{pre}_elicit_to_tell"] = v2_agg._ratio(el, te)
        fc = vals[ix["FEEDBACK_CORRECT"]] if "FEEDBACK_CORRECT" in ix else 0.0
        fi = vals[ix["FEEDBACK_INCORRECT"]] if "FEEDBACK_INCORRECT" in ix else 0.0
        out[f"{fam}|{pre}_feedback_positive_ratio"] = float(fc / (fc + fi + v2_agg.EPS))

    # --- rate: mean over all turns of that role ------------------------------------------
    rT = v2_agg.rates(T, tcols, mask=tmask)
    rS = v2_agg.rates(S, scols, mask=smask)
    emit("rate", "v2t", rT, tcols)
    ratios("rate", "v2t", rT, tcols)
    emit("rate", "v2s", rS, scols)

    # --- shrunk: same columns, noise made independent of transcript length ---------------
    # A rate's sampling sd is sigma/sqrt(n), so plain rates are noisiest on short
    # transcripts. See `v2_agg.shrunk_rates`.
    sT = v2_agg.shrunk_rates(T, tcols, mask=tmask)
    sS = v2_agg.shrunk_rates(S, scols, mask=smask)
    emit("shrunk", "v2ht", sT, tcols)
    ratios("shrunk", "v2ht", sT, tcols)
    emit("shrunk", "v2hs", sS, scols)

    # --- win{K}: same columns, aggregated over the last K turns (length-invariant) --------
    for _k, _pre in ((12, "v2wa"), (24, "v2wb"), (48, "v2wc")):
        _wT = v2_agg.window_rates(T, tcols, mask=tmask, k=_k)
        _wS = v2_agg.window_rates(S, scols, mask=smask, k=_k)
        emit(f"win{_k}", f"{_pre}t", _wT, tcols)
        ratios(f"win{_k}", f"{_pre}t", _wT, tcols)
        emit(f"win{_k}", f"{_pre}s", _wS, scols)

    # --- shrunkeb: per-code empirical-Bayes kappa ---------------------------------------
    # k_c = sigma2_within,c / Var_between,c, by method of moments on the training data only
    # (`v2_agg.CODE_KAPPA`).
    ebT = v2_agg.eb_shrunk_rates(T, tcols, mask=tmask)
    ebS = v2_agg.eb_shrunk_rates(S, scols, mask=smask)
    emit("shrunkeb", "v2et", ebT, tcols); ratios("shrunkeb", "v2et", ebT, tcols)
    emit("shrunkeb", "v2es", ebS, scols)

    # --- shrunkk35: global kappa=35 ------------------------------------------------------
    gT = v2_agg.shrunk_rates(T, tcols, mask=tmask, kappa=35.0)
    gS = v2_agg.shrunk_rates(S, scols, mask=smask, kappa=35.0)
    emit("shrunkk35", "v2gt", gT, tcols); ratios("shrunkk35", "v2gt", gT, tcols)
    emit("shrunkk35", "v2gs", gS, scols)

    # --- shrunkq: shrunk rates + bounded share ratios ------------------------------------
    # `v2qt_elicit_share` is `el/(el+te)`, a bounded [0, 1] form of the capped quotient
    # `elicit_to_tell`.
    emit("shrunkq", "v2qt", sT, tcols)
    _ix = {c: j for j, c in enumerate(tcols)}
    _el = sum(sT[_ix[c]] for c in v2_agg.ELICIT if c in _ix)
    _te = sum(sT[_ix[c]] for c in v2_agg.TELL if c in _ix)
    out["shrunkq|v2qt_elicit_share"] = float(_el / (_el + _te + v2_agg.EPS))
    _fc = sT[_ix["FEEDBACK_CORRECT"]] if "FEEDBACK_CORRECT" in _ix else 0.0
    _fi = sT[_ix["FEEDBACK_INCORRECT"]] if "FEEDBACK_INCORRECT" in _ix else 0.0
    out["shrunkq|v2qt_feedback_positive_ratio"] = float(_fc / (_fc + _fi + v2_agg.EPS))
    emit("shrunkq", "v2qs", sS, scols)

    # --- instr: same columns, instructional denominator ---------------------------------
    # Residual codes keep the plain denominator: "what share of this session was small talk"
    # is a real session property, and dividing it by its own complement would be circular.
    iT = v2_agg.rates(T, tcols, weights=wT, mask=tmask)
    iS = v2_agg.rates(S, scols, weights=wS, mask=smask)
    for j, c in enumerate(tcols):
        if c in v2_agg.TUTOR_RESID:
            iT[j] = rT[j]
    for j, c in enumerate(scols):
        if c in v2_agg.STUDENT_RESID:
            iS[j] = rS[j]
    emit("instr", "v2ti", iT, tcols)
    ratios("instr", "v2ti", iT, tcols)
    emit("instr", "v2si", iS, scols)

    # --- trim: same columns, boundary set by the labels ---------------------------------
    lo, hi = v2_agg.instructional_span(wT)
    span = np.zeros(len(wT), dtype=bool)
    span[lo:hi + 1] = True
    tT = v2_agg.rates(T, tcols, mask=tmask & span)
    tS = v2_agg.rates(S, scols, mask=smask & span)
    emit("trim", "v2tt", tT, tcols)
    ratios("trim", "v2tt", tT, tcols)
    emit("trim", "v2st", tS, scols)

    # --- bound / slope / disp / trans ----------------------------------------------------
    for k, v in v2_agg.boundary_features(wT, wS, T, tcols, "v2b").items():
        if k.endswith("_code_mass"):
            continue          # exact linear combination of the rates (see boundary_features)
        out[f"bound|{k}"] = v
    out["bound|v2b_stu_instr_share"] = float(wS.mean()) if len(wS) else np.nan

    for k, v in v2_agg.slope_features(T, tcols, AXES_T, "v2sl").items():
        out[f"slope|{k}"] = v
    for k, v in v2_agg.slope_features(S, scols, AXES_S, "v2sl_s").items():
        out[f"slope|{k}"] = v

    for k, v in v2_agg.dispersion_features(T, tcols, AXES_T, "v2d").items():
        out[f"disp|{k}"] = v

    for k, v in v2_agg.transition_features(T, tcols, S, scols, "v2x").items():
        out[f"trans|{k}"] = v

    for k, v in v2_agg.ratio_features(T, tcols, "v2r").items():
        out[f"ratio|{k}"] = v

    for k, v in v2_agg.quality_features(T, tcols, S, scols, tmask, "v2q").items():
        if k.endswith("_entropy"):
            continue          # measures length, not register (see quality_features)
        out[f"qual|{k}"] = v
    return out


def main() -> None:
    """Build all families for every session that has both pools.

    Returns:
        None. Writes one artifacts/v2fam_{family}_{tag}.csv per session-keyed family and,
        unless --skip-lowin, the response-keyed artifacts/v2fam_lowin_{tag}.csv.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor", default="artifacts/v2t_turn_probs_v2ext_122b.pkl.gz")
    ap.add_argument("--student", default="artifacts/v2s_turn_probs_v2stu_122b.pkl.gz")
    ap.add_argument("--tag", default="122b")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only N sessions")
    ap.add_argument("--skip-lowin", action="store_true",
                    help="session-keyed families only; skips reading the transcripts")
    args = ap.parse_args()

    t0 = time.time()
    tp = pd.read_pickle(ROOT / args.tutor, compression="gzip")
    sp = pd.read_pickle(ROOT / args.student, compression="gzip")
    # Column order comes from the input table, never from a constant here: a v2 run with a
    # different code inventory must not silently misalign against a booster fitted on another.
    tcols = [c for c in tp.columns
             if c not in ("session_id", "turn", "pos", "n_turns", "_labelled")]
    scols = [c for c in sp.columns
             if c not in ("session_id", "turn", "pos", "n_turns", "_labelled")]
    print(f"tutor {tp.shape} / {len(tcols)} codes; student {sp.shape} / {len(scols)} codes "
          f"({time.time() - t0:.0f}s)", flush=True)

    tg = {s: g for s, g in tp.groupby("session_id", sort=False)}
    sg = {s: g for s, g in sp.groupby("session_id", sort=False)}
    sessions = [s for s in sg if s in tg]
    if args.limit:
        sessions = sessions[:args.limit]
    print(f"{len(sessions):,} sessions with both pools ({time.time() - t0:.0f}s)", flush=True)

    train = data.load_train()
    resp = train[["response_id", "session_id", "learning_objective"]]

    rows, lowin_rows = [], []
    store = data.default_store()
    need_lo = not args.skip_lowin
    by_sess = {s: g for s, g in resp.groupby("session_id", sort=False)} if need_lo else {}

    def _stream():
        """Yield per-turn word sets from the raw transcripts, in `sessions` order.

        Not from `turn_text.pkl.gz`: that table truncates every field to 220 characters, so
        keyword matching on it would see a different word set than inference sees on the full
        transcript.

        Uses `store.iter`, not `store.load`: `load` re-opens the zip on every call and so
        re-parses the central directory of a 22,821-member archive once per session, measured
        at ~0.4 s per session (2.5 h for the corpus) against ~2 min with one open handle. The
        other builders do the same.

        Yields:
            ``(session_id, [word set per paired turn])``.
        """
        for sid, tdf in store.iter(sessions):
            try:
                turns, _ = build_turns(tdf)
            except Exception:
                turns = []
            yield sid, [_lo_words_text(t) for t in turns]

    pending = _stream() if need_lo else iter(())

    for k, sid in enumerate(sessions):
        if k % 2000 == 0:
            print(f"  {k:,}/{len(sessions):,}  {time.time() - t0:.0f}s", flush=True)
        T, S, tmask, smask, pos = _session_arrays(tg[sid], sg[sid], tcols, scols)
        r = session_families(T, S, tmask, smask, tcols, scols)
        r["session_id"] = sid
        rows.append(r)

        if not need_lo:
            continue
        wsid, words = next(pending)
        assert wsid == sid, f"prefetch desynchronised: {wsid} != {sid}"
        if not words or sid not in by_sess:
            continue
        for rid, lo_text in zip(by_sess[sid].response_id, by_sess[sid].learning_objective):
            hits = _hits(words, lo_text, len(pos), pos)
            win = None
            if hits is not None:
                win = _dilate(hits, LO_PAD_TURNS)
                win = win if win.any() else None
            f = v2_agg.lo_window_features(T, tcols, S, scols, win, "v2w")
            f.update(v2_agg.lo_soft_coverage(hits, len(pos), "v2w"))
            f["response_id"] = rid
            lowin_rows.append(f)

    fam = pd.DataFrame(rows)
    for name in sorted({c.split("|", 1)[0] for c in fam.columns if "|" in c}):
        cols = [c for c in fam.columns if c.startswith(f"{name}|")]
        out = fam[["session_id"] + cols].rename(
            columns={c: c.split("|", 1)[1] for c in cols})
        p = ROOT / f"artifacts/v2fam_{name}_{args.tag}.csv"
        out.to_csv(p, index=False)
        print(f"wrote {p.name}  {len(out):,} sessions x {len(cols)} cols", flush=True)

    if lowin_rows:
        lw = pd.DataFrame(lowin_rows)
        lw = lw[["response_id"] + [c for c in lw.columns if c != "response_id"]]
        p = ROOT / f"artifacts/v2fam_lowin_{args.tag}.csv"
        lw.to_csv(p, index=False)
        cov = lw.drop(columns="response_id").notna().any(axis=1).mean()
        print(f"wrote {p.name}  {len(lw):,} responses x {lw.shape[1] - 1} cols, "
              f"{cov:.1%} with a window", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------
import re  # noqa: E402

_W = re.compile(r"[A-Za-z']+")


def _lo_words_text(t: dict) -> set:
    """Return the lower-cased word set of one paired turn.

    Both the tutor and the student text are used: either party can name an objective, and
    the window is about the topic, not about who raised it.
    """
    return {w.lower() for w in _W.findall(f"{t['teacher']} {t['student']}")}


def _hits(words: list, lo_text, n: int, pos: np.ndarray):
    """Mark the turns on the aligned turn axis that name this objective.

    The keyword rule is imported from `histpos_features._lo_words`, the same rule
    `features._lo_transcript_stats` uses, not a re-typed copy. Kept separate from the
    dilation so that the hard window and the soft kernel are built from the same hits.

    Args:
        words: word set per paired turn of the transcript, from `_lo_words_text`.
        lo_text: the objective text.
        n: length of the aligned turn axis.
        pos: turn positions of the aligned axis (indices into the transcript's turns).

    Returns:
        Boolean array of length ``n``, or None when the objective has no content words or
        is never mentioned. The caller turns None into NaN rather than zero, because "never
        discussed" and "discussed with no contrast" are different states.
    """
    lw = _lo_words(lo_text)
    if not lw:
        return None
    raw = np.array([bool(lw & ws) for ws in words])
    if not raw.any():
        return None
    # `pos` indexes into `turns`, so it maps the aligned axis back onto the transcript.
    keep = pos[pos < len(words)]
    h = np.zeros(n, dtype=bool)
    h[:len(keep)] = raw[keep]
    return h if h.any() else None


def _dilate(hits: np.ndarray, pad: int) -> np.ndarray:
    """Return the hard +/-``pad`` window around every hit, as a boolean array."""
    w = np.zeros(len(hits), dtype=bool)
    for i in np.where(hits)[0]:
        w[max(0, i - pad): min(len(hits), i + pad + 1)] = True
    return w


if __name__ == "__main__":
    main()
