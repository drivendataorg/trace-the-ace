#!/usr/bin/env python
"""Distil a tutoring-move annotation into a CPU classifier and build its session-level block.

The fourth use of the recipe of distil_turn_correctness.py, distil_nto_moves.py and
distil_student_states.py: TF-IDF (word + char) -> one binary logistic head per family,
session-disjoint folds, soft probabilities averaged to session level. The shipped model uses
it for the taxonomy-v2 tutor and student moves (codebook:
src/training/docs/39_moves_annotations_v2.md; labels from Qwen3.5-122B, step 02e).

The script reads two label formats (`(session_id, turn, codes)` with `|`-joined multi-labels,
and the taxonomy-v2 `(session_id, turn, code, code_2, aux)`), normalises them to the same
per-turn code set, and runs them through one code path with one classifier configuration
(imported from distil_nto_moves).

`--restrict-sessions` restricts the labels to a fixed session list; the shipped runs pass
artifacts/v2_sessions_3072.txt.

The tutor map (--map) selects the families:

  `v2_ext`    every code its own family (the shipped tutor block).
  `student`   every student code its own family (the shipped student block).
  `v1_19`, `v2_cross`  the 19 families of `build_nto_rate_features.FAMILIES` (`v2_cross` via
              `V2_CROSSWALK`).

In:  --labels CSVs, artifacts/turn_text.pkl.gz (or --turn-text), optional
     --restrict-sessions list
Out: artifacts/{prefix}_clf_{tag}.joblib (all-data fit; v2t_clf_v2ext_122bC and
     v2s_clf_v2stu_122bC ship in assets/ as v2t_clf.joblib and v2s_clf.joblib),
     artifacts/{prefix}_turn_probs_{tag}.pkl.gz, artifacts/{prefix}_features_{tag}.csv,
     artifacts/{prefix}_features_{tag}.meta.json, figures/data/{prefix}_distil_{tag}.md

Usage (reproduce.py step 03f, run from build/workspace/ with single-threaded BLAS,
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; tutor, then student):
    python scripts/distil_v2_moves.py --labels artifacts/v2labels/122b/moves_v2_ext_tutor_shard0.csv
        --map v2_ext --role tutor --prefix v2t --tag v2ext_122bC
        --restrict-sessions artifacts/v2_sessions_3072.txt --ratios v2
    python scripts/distil_v2_moves.py --labels artifacts/v2labels/122b/moves_v2_ext_student_shard0.csv
        --map student --role student --prefix v2s --tag v2stu_122bC
        --restrict-sessions artifacts/v2_sessions_3072.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.metrics import roc_auc_score  # noqa: E402
from magnificat.vocab_order import dump_sorted  # noqa: E402

from distil_nto_moves import CHUNK, MIN_POS, N_FOLDS, head, vectorizer  # noqa: E402
from build_nto_rate_features import FAMILIES as V1_FAMILIES  # noqa: E402

FIT_CAP = 250_000

#: Hard cap on `elicit_to_tell`. It is a constant because deriving it from the data being
#: scored would make the feature depend on the other samples in the batch, which the
#: competition rules forbid.
ELICIT_TELL_CAP = 10.0

# ---------------------------------------------------------------------------------------
# Taxonomy-v2 code -> family of `build_nto_rate_features.FAMILIES`, for `--map v2_cross`.
# Only near-exact twins are mapped.
# ---------------------------------------------------------------------------------------
V2_CROSSWALK = {
    "FEEDBACK_CORRECT": "fb_correct",
    "FEEDBACK_INCORRECT": "fb_incorrect",
    "FEEDBACK_NEUTRAL": "fb_neutral",
    "GIVING_PRAISE": "praise",
    "TUTOR_SOCIAL": "rapport",
    "PROMPTING_SELF_EXPLANATION": "prompt_self_expl",
    "PROMPTING_NEXT_STEP": "prompt_next_step",
    "PROMPTING_SELF_CORRECTION": "prompt_self_correct",
    "REVOICING": "revoicing",
    "GIVING_HINT": "hint",
    "EXPLAINING_CONCEPTUAL": "explain_conceptual",
    "EXPLAINING_PROCEDURAL": "explain_procedural",
    "GIVING_ANSWER": "give_answer",
    "TUTOR_TECHNICAL": "technical",
}

#: (elicit codes, tell codes, positive-feedback code, negative-feedback code) for the two
#: derived ratios, per label format (--ratios; the shipped tutor run passes `v2`).
RATIO_SPEC = {
    "v1": (["prompt_self_expl", "prompt_next_step", "probe_understand", "probe_prior"],
           ["explain_conceptual", "explain_procedural", "give_answer"],
           "fb_correct", "fb_incorrect"),
    "v2": (["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP",
            "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
            "PROMPTING_SELF_CORRECTION", "RELEASE_HANDOFF"],
           ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
            "WORKED_DEMONSTRATION"],
           "FEEDBACK_CORRECT", "FEEDBACK_INCORRECT"),
}


def load_labels(paths: list[Path]) -> pd.DataFrame:
    """Read label CSVs of either format into one per-turn code-set table.

    Args:
        paths: label CSVs; a file with a `codes` column is read as `|`-joined codes, any
            other file as `code` plus optional `code_2` (the taxonomy-v2 format).

    Returns:
        DataFrame with columns session_id, turn, codes (a set of code strings), one row per
        (session_id, turn) (first occurrence kept).
    """
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        if "codes" in d.columns:                       # `|`-joined codes
            d["codes"] = d["codes"].fillna("").apply(
                lambda s: {c.strip() for c in str(s).split("|") if c.strip()})
        else:                                          # taxonomy-v2 format
            c2 = d["code_2"].fillna("") if "code_2" in d.columns else ""
            d["codes"] = [({a} | ({b} if b else set()))
                          for a, b in zip(d["code"].fillna(""), c2)]
            d["codes"] = d["codes"].apply(lambda s: {c for c in s if c})
        frames.append(d[["session_id", "turn", "codes"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["session_id", "turn"])


def family_map(kind: str, codes_seen: set[str]) -> tuple[dict[str, str], list[str]]:
    """Return the code-to-family mapping and the ordered family list for one --map value.

    Args:
        kind: "v1_19", "v2_cross", "v2_ext" or "student".
        codes_seen: codes present in the labels (defines the families of v2_ext / student).

    Returns:
        (code -> family dict, sorted list of family names).
    """
    if kind == "v1_19":
        return ({c: f for f, cs in V1_FAMILIES.items() for c in cs}, sorted(V1_FAMILIES))
    if kind == "v2_cross":
        return dict(V2_CROSSWALK), sorted(V1_FAMILIES)   # the 19 columns of `v1_19`
    if kind == "v2_ext":
        fams = sorted(codes_seen)
        return {c: c for c in fams}, fams
    if kind == "student":
        fams = sorted(codes_seen)
        return {c: c for c in fams}, fams
    raise SystemExit(f"unknown --map {kind!r}")


def stable_folds(sessions: np.ndarray, n: int = N_FOLDS) -> np.ndarray:
    """Session-disjoint folds from a stable hash of `session_id`.

    Not `GroupKFold`: numpy's sort dispatch differs between Haswell and Skylake CPUs, so
    GroupKFold's tie-breaking among equal-sized groups depends on the machine. blake2b gives
    the same folds on every machine.

    Args:
        sessions: session_id per row.
        n: number of folds.

    Returns:
        Integer fold index in [0, n) per row.
    """
    import hashlib
    return np.array([int(hashlib.blake2b(str(s).encode(), digest_size=8).hexdigest(), 16) % n
                     for s in sessions])


def build_text(df: pd.DataFrame, role: str) -> np.ndarray:
    """Return the input text of each turn, exactly as the NTO and student-state classifiers
    build it.

    tutor:   prev student turn + " [T] " + tutor utterance   (as in distil_nto_moves)
    student: tutor utterance    + " [S] " + student turn     (`distil_student_states._text`)

    Do not change it: the inference copy (`magnificat.v2_features`) must reproduce it.
    Both sides are available at inference. The truncations are the ones
    build_evidence_features.py applied when writing `turn_text.pkl.gz`.

    Args:
        df: per-turn table sorted by (session_id, turn), with tutor_q and student columns.
        role: "tutor" or "student".

    Returns:
        Array of input strings, one per row of `df`.
    """
    if role == "tutor":
        prev = df.groupby("session_id", sort=False)["student"].shift(1).fillna("")
        return (prev.str[:220] + " [T] " + df.tutor_q.fillna("").str[-220:]).values
    return (df.tutor_q.fillna("").str[-200:] + " [S] "
            + df.student.fillna("").str[:300]).values


def main() -> None:
    """Distil one label set: out-of-fold heads, all-data classifier, whole-corpus per-turn
    table, session block, a train/serve parity check and the report.

    Returns:
        None. Writes artifacts/{prefix}_clf_{tag}.joblib,
        artifacts/{prefix}_turn_probs_{tag}.pkl.gz, artifacts/{prefix}_features_{tag}.csv,
        artifacts/{prefix}_features_{tag}.meta.json and
        figures/data/{prefix}_distil_{tag}.md. If `magnificat.v2_features` does not
        reproduce the session block within 1e-4, the classifier file is deleted and the
        script exits with an error before writing the report.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="label CSVs (see load_labels)")
    ap.add_argument("--map", required=True,
                    choices=["v1_19", "v2_cross", "v2_ext", "student"])
    ap.add_argument("--role", default="tutor", choices=["tutor", "student"])
    ap.add_argument("--prefix", default="ntodst", help="feature column prefix")
    ap.add_argument("--tag", required=True, help="suffix for every output path")
    ap.add_argument("--restrict-sessions", default=None,
                    help="file of newline-separated session_ids to restrict the labels to "
                         "(the shipped runs pass artifacts/v2_sessions_3072.txt)")
    ap.add_argument("--fit-cap", type=int, default=FIT_CAP)
    ap.add_argument("--turn-text", default="turn_text.pkl.gz",
                    help="alternate per-turn table, e.g. one built with a wider --trunc. The "
                         "classifier's vectoriser is fitted on whatever this holds, so a "
                         "classifier fitted on another table is not interchangeable with the "
                         "shipped one.")
    ap.add_argument("--trunc", type=int, nargs=2, default=(220, 220), metavar=("TEACHER","STUDENT"),
                    help="the truncation this table was built with; recorded in the "
                         "classifier file so that the inference code reproduces it rather "
                         "than assuming 220.")
    ap.add_argument("--ratios", default=None, choices=["v1", "v2"],
                    help="also emit elicit_to_tell + feedback_positive_ratio")
    args = ap.parse_args()

    tt = pd.read_pickle(ROOT / "artifacts" / args.turn_text, compression="gzip")
    lab = load_labels([Path(p) for p in args.labels])
    print(f"labels: {len(lab):,} turns / {lab.session_id.nunique():,} sessions", flush=True)

    if args.restrict_sessions:
        keep = {s.strip() for s in Path(args.restrict_sessions).read_text().split() if s.strip()}
        lab = lab[lab.session_id.isin(keep)]
        print(f"  coverage-matched to {len(keep):,} sessions -> {len(lab):,} turns "
              f"/ {lab.session_id.nunique():,} sessions", flush=True)

    df = tt.merge(lab, on=["session_id", "turn"], how="inner")
    df = df.sort_values(["session_id", "turn"]).reset_index(drop=True)
    # The student pass codes the student slot of the turn; turns with no student text carry
    # no codeable content and would train the head on an empty string.
    if args.role == "student":
        df = df[df.student.fillna("").str.strip() != ""].reset_index(drop=True)
    print(f"coded turns joined to turn_text: {len(df):,} / "
          f"{df.session_id.nunique():,} sessions", flush=True)

    seen = set().union(*df.codes) if len(df) else set()
    code_of, fams = family_map(args.map, seen)
    unmapped = seen - set(code_of)
    if unmapped:
        # Expected under v2_cross, which maps only part of the codes (see V2_CROSSWALK).
        # Under the other maps it means the codebook has drifted, so it is flagged.
        print(f"  {len(unmapped)} codes not in the map"
              f"{' (expected for v2_cross)' if args.map == 'v2_cross' else ' (WARNING)'}: "
              f"{sorted(unmapped)[:10]}", flush=True)

    hit = df.codes.apply(lambda cs: {code_of[c] for c in cs if c in code_of})
    Y = pd.DataFrame({f: hit.apply(lambda s, f=f: int(f in s)).values for f in fams})
    print("family base rates:\n" + Y.mean().round(4).to_string(), flush=True)

    text = build_text(df, args.role)
    groups = df.session_id.values
    oof = pd.DataFrame(index=df.index, columns=fams, dtype=float)
    rng = np.random.default_rng(221)

    fold = stable_folds(groups)
    print(f"classifier folds (blake2b on session_id): "
          f"{np.bincount(fold, minlength=N_FOLDS).tolist()} turns per fold", flush=True)
    for k in range(N_FOLDS):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        t0 = time.time()
        fit_idx = (rng.choice(tr, args.fit_cap, replace=False)
                   if args.fit_cap and len(tr) > args.fit_cap else tr)
        vec = vectorizer(None)
        Xtr = vec.fit_transform(text[fit_idx])
        Xte = vec.transform(text[te])
        for f in fams:
            ytr = Y[f].values[fit_idx]
            if ytr.sum() < MIN_POS:
                oof.loc[te, f] = float(Y[f].values[tr].mean())
                continue
            m = head(None)
            m.fit(Xtr, ytr)
            oof.loc[te, f] = np.concatenate(
                [m.predict_proba(Xte[i:i + CHUNK])[:, 1]
                 for i in range(0, Xte.shape[0], CHUNK)])
        print(f"  fold {k + 1}/{N_FOLDS}: {Xtr.shape[0]:,}x{Xtr.shape[1]:,} "
              f"in {time.time() - t0:.0f}s", flush=True)
        del vec, Xtr, Xte

    aucs = {f: (roc_auc_score(Y[f], oof[f]) if Y[f].nunique() > 1 and oof[f].nunique() > 1
                else float("nan")) for f in fams}
    mean_auc = float(np.nanmean(list(aucs.values())))
    print(f"\nmean OOF AUC {mean_auc:.4f} over {len(fams)} heads", flush=True)

    # ---- all-data classifier, and per-turn probabilities for the whole corpus ----------
    # The annotation covers only part of the sessions. As in distil_student_states.py, a
    # labelled session keeps its OOF value and every other turn is scored by the all-data
    # classifier, so the block also covers the unannotated sessions. Both match the
    # inference condition ("a session this annotator has not seen"); using the all-data
    # classifier on a labelled session would be optimistic.
    t0 = time.time()
    fit_idx = (rng.choice(len(df), args.fit_cap, replace=False)
               if args.fit_cap and len(df) > args.fit_cap else np.arange(len(df)))
    vec = vectorizer(None)
    Xall = vec.fit_transform(text[fit_idx])
    heads, base = {}, {}
    for f in fams:
        yf = Y[f].values[fit_idx]
        base[f] = float(Y[f].mean())
        heads[f] = head(None).fit(Xall, yf) if yf.sum() >= MIN_POS else None
    print(f"all-data classifier fitted on {len(fit_idx):,} turns in {time.time() - t0:.0f}s",
          flush=True)
    # Save the classifier. Inference does not read the session CSVs: it rebuilds the block
    # from the transcripts with this saved classifier, as `nto_features.build_nto_matrix`
    # does with `nto_moves_clf.joblib`.
    import joblib
    p_clf = ROOT / f"artifacts/{args.prefix}_clf_{args.tag}.joblib"
    dump_sorted({"vec": vec, "heads": heads, "base": base, "families": fams,
                 "role": args.role, "prefix": args.prefix, "map": args.map,
                 # Record the truncation this classifier was fitted under, so the inference
                 # copy can assert against it instead of re-deriving it.
                 # `turn_text.pkl.gz` truncates on write (teacher[-220:], student[:220]) and
                 # build_text truncates again; the composition is what the vectoriser saw.
                 "trunc": {"table": max(args.trunc), "tutor_ctx": -200, "tutor_utt": -220,
                           "student_ctx": 220, "student_utt": 300}},
               p_clf, compress=3)
    print(f"wrote {p_clf.name} ({p_clf.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Score the same population the annotator was asked to code: a tutor pass codes turns
    # with a tutor utterance, a student pass turns with student text. Scoring the others
    # would feed the vectoriser an empty string it was never fitted on.
    full = tt.sort_values(["session_id", "turn"]).reset_index(drop=True)
    slot = "tutor_q" if args.role == "tutor" else "student"
    full = full[full[slot].fillna("").str.strip() != ""].reset_index(drop=True)
    ftext = build_text(full, args.role)
    lab_keys = set(zip(df.session_id, df.turn))
    is_lab = pd.Series(list(zip(full.session_id, full.turn))).isin(lab_keys).to_numpy()
    print(f"scoring {int((~is_lab).sum()):,} unlabelled turns "
          f"({is_lab.mean():.1%} of the corpus already has OOF)", flush=True)

    P = pd.DataFrame(index=full.index, columns=fams, dtype=np.float32)
    un = np.where(~is_lab)[0]
    for i in range(0, len(un), CHUNK):
        sl = un[i:i + CHUNK]
        Xc = vec.transform(ftext[sl])
        for f in fams:
            P.iloc[sl, P.columns.get_loc(f)] = (
                heads[f].predict_proba(Xc)[:, 1].astype(np.float32) if heads[f] is not None
                else np.float32(base[f]))
    src = oof.set_index(pd.MultiIndex.from_arrays([df.session_id, df.turn]))
    tgt = pd.MultiIndex.from_arrays([full.session_id[is_lab], full.turn[is_lab]])
    for f in fams:
        P.loc[is_lab, f] = src[f].reindex(tgt).to_numpy(dtype=np.float32)

    turns = full[["session_id", "turn", "pos", "n_turns"]].copy()
    for f in fams:
        turns[f] = P[f].to_numpy(np.float32)
    turns["_labelled"] = is_lab
    p_turn = ROOT / f"artifacts/{args.prefix}_turn_probs_{args.tag}.pkl.gz"
    turns.to_pickle(p_turn, compression="gzip")

    S = P.assign(session_id=full.session_id.values).groupby("session_id").mean()
    S.columns = [f"{args.prefix}_{c.lower()}_rate" for c in S.columns]
    if args.ratios:
        el, te_, fc, fi = RATIO_SPEC[args.ratios]
        col = lambda c: f"{args.prefix}_{c.lower()}_rate"  # noqa: E731
        have = lambda cs: [col(c) for c in cs if col(c) in S.columns]  # noqa: E731
        r = S[have(el)].sum(axis=1) / (S[have(te_)].sum(axis=1) + 1e-3)
        # Clip at a hardcoded constant, never at a quantile of the data being scored (that
        # would be a cross-sample statistic, which the competition rules forbid at inference).
        #
        # Unlike `feedback_positive_ratio`, whose numerator is part of its denominator so it
        # is bounded in [0, 1], `elicit_to_tell` is unbounded and its tail is set by the 1e-3
        # guard.
        S[f"{args.prefix}_elicit_to_tell"] = r.clip(upper=ELICIT_TELL_CAP)
        # numerator is part of the denominator, so this one is already bounded in [0, 1]
        S[f"{args.prefix}_feedback_positive_ratio"] = (
            S[col(fc)] / (S[col(fc)] + S[col(fi)] + 1e-3))
        n_clip = int((r > ELICIT_TELL_CAP).sum())
        print(f"elicit_to_tell: clipped {n_clip:,}/{len(r):,} sessions "
              f"({n_clip / max(1, len(r)):.3%}) at {ELICIT_TELL_CAP}; "
              f"pre-clip max {r.max():.2f}", flush=True)
    out = S.reset_index()
    p_dst = ROOT / f"artifacts/{args.prefix}_features_{args.tag}.csv"
    out.to_csv(p_dst, index=False)
    # Provenance sidecar: the labels, map, role and training coverage of this run.
    import json
    (ROOT / f"artifacts/{args.prefix}_features_{args.tag}.meta.json").write_text(json.dumps({
        "tag": args.tag, "map": args.map, "role": args.role,
        "labelled_sessions": int(df.session_id.nunique()),
        "labelled_turns": int(len(df)), "corpus_turns_scored": int(len(full)),
        "child_mean_oof_auc": mean_auc, "n_families": len(fams),
        "labels": [Path(p).name for p in args.labels],
    }, indent=2))
    print(f"wrote {p_dst.name} ({len(out):,} sessions x {out.shape[1] - 1} features)",
          flush=True)

    # ---- train/serve parity check before the classifier can be used --------------------
    # Train/serve skew is invisible in-domain: training reads the already-truncated table,
    # so an inference copy that truncates differently scores perfectly here and wrong only
    # at inference. So the classifier is deleted unless `magnificat.v2_features` reproduces
    # the block on 150 unlabelled sessions.
    try:
        from magnificat import v2_features as _v2f
        _lab = set(df.session_id.unique())
        _pool = [s for s in out.session_id if s not in _lab][:150]
        if _pool:
            _f = pd.DataFrame({"session_id": _pool,
                               "response_id": [f"r{i}" for i in range(len(_pool))]})
            _g = _v2f.build_v2_matrix(_f, clf=joblib.load(p_clf))
            _g.index = _pool
            _cols = [c for c in _v2f.block_cols(joblib.load(p_clf)) if c in out.columns]
            _r = out.set_index("session_id").loc[_pool, _cols]
            _d = float((_g[_cols] - _r).abs().to_numpy().max())
            _tol = 1e-4          # float32 storage floor is ~1.2e-07; a real mismatch is ~1e-2
            print(f"train/serve parity: max |Δ| {_d:.3e} on {len(_pool)} unlabelled sessions "
                  f"(tol {_tol:.0e})", flush=True)
            if _d > _tol:
                p_clf.unlink(missing_ok=True)
                raise SystemExit(
                    f"PARITY FAILED ({_d:.3e} > {_tol:.0e}): magnificat.v2_features does not "
                    f"reproduce what this run fitted, so the classifier file was deleted; a "
                    f"score computed with it on another corpus would measure train/serve skew, "
                    f"not transfer. Fix the text construction in magnificat.v2_features and "
                    f"re-run.")
    except ImportError:
        print("WARNING: magnificat.v2_features unavailable; train/serve parity not verified",
              flush=True)

    L = [f"# Distilled move child — `{args.tag}` ({args.map}, role={args.role})", "",
         f"- labels: {', '.join(Path(p).name for p in args.labels)}",
         f"- coded turns: {len(df):,} / {df.session_id.nunique():,} sessions",
         f"- corpus turns scored: {len(full):,} ({is_lab.mean():.1%} OOF, "
         f"{1 - is_lab.mean():.1%} all-data child)",
         f"- child mean OOF AUC: **{mean_auc:.4f}** over {len(fams)} heads",
         "", "Child AUC measures agreement with the annotator labels, not the effect on the "
         "XGBoost model.", "",
         "| family | OOF AUC | base rate |", "|---|---|---|"]
    for f in fams:
        L.append(f"| `{f}` | {aucs[f]:.4f} | {Y[f].mean():.4f} |")
    p_md = ROOT / f"figures/data/{args.prefix}_distil_{args.tag}.md"
    p_md.write_text("\n".join(L) + "\n")
    print(f"wrote {p_md.name}", flush=True)


if __name__ == "__main__":
    main()
