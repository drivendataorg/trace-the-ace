#!/usr/bin/env python
r"""Reproducible EDA (training stage 1): every number and figure in outputs/eda/REPORT.md.

Also writes the tables later stages read: transcript_stats.csv (cached; --force recomputes),
objective_overlap.csv, objective_topics.csv, cv_folds.csv and baseline_oof.csv.
--skip-baseline skips the TF-IDF logistic-regression CV/leakage experiment, the slowest step.

    python src/ace/eda/run_eda.py --data-dir data --transcripts-dir data/train_transcripts \
        --out outputs/eda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42

# ---------------------------------------------------------------- palette ---
P = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7",
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
}

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": P["surface"], "axes.facecolor": P["surface"],
        "savefig.facecolor": P["surface"], "figure.dpi": 150,
        "text.color": P["ink"], "axes.labelcolor": P["ink2"],
        "xtick.color": P["muted"], "ytick.color": P["muted"],
        "axes.edgecolor": P["baseline"], "axes.linewidth": 1.0,
        "grid.color": P["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 9,
    })
    return plt


# ---------------------------------------------------- transcript stats pass ---
STOPWORDS = set("""a an and are as at be been being by can could do does did for from
has have had he her him his how i if in into is it its me my not of on or our so
she that the their them then there they this to up us was we were what when which
who will with would you your yours it's don't i'm that's""".split())

BRACKET_RE = re.compile(r"\[([a-z][a-z ]{0,30})\]")
TOKEN_RE = re.compile(r"[a-z']+")
GREET_RE = re.compile(r"\b(?:hi|hello|hey|morning|afternoon),?\s+([A-Z][a-z]{2,})")
NOT_NAMES = {"There", "How", "Good", "Guys", "Everyone", "Miss", "Sir", "Thank",
             "Thanks", "Well", "Yes", "And", "You", "Are", "The", "This", "What"}

_WORKER_OBJ: dict[str, list[tuple[str, frozenset]]] = {}

def _init_worker(obj_map):
    global _WORKER_OBJ
    _WORKER_OBJ = obj_map


def parse_timestamp(ts: str):
    try:
        h, m, s = ts.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def parse_transcript(path_str: str):
    """One pass over a transcript CSV -> stats dict, bracket Counter, role Counter,
    per-(session, objective) lexical overlap."""
    import csv
    path = Path(path_str)
    session_id = path.stem
    st = dict(session_id=session_id, n_utt=0, n_tutor=0, n_student=0, n_background=0,
              n_other_role=0, w_total=0, w_student=0, w_tutor=0, c_total=0,
              n_unclear=0, n_bracket=0, n_empty=0, n_ts_bad=0, ts_nonmono=0,
              tutor_q=0, student_q=0, max_utt_id=-1, parse_error=0)
    brackets: Counter = Counter()
    roles: Counter = Counter()
    tok_set: set[str] = set()
    md5 = hashlib.md5()
    first_ts = last_ts = None
    prev_ts = None
    greet_name = ""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                role = (row.get("role") or "").strip()
                content = row.get("content") or ""
                roles[role] += 1
                st["n_utt"] += 1
                try:
                    uid = int(row.get("utterance_id") or -1)
                    st["max_utt_id"] = max(st["max_utt_id"], uid)
                except ValueError:
                    pass
                if not content.strip():
                    st["n_empty"] += 1
                md5.update(content.encode("utf-8", "replace"))
                low = content.lower()
                toks = low.split()
                nw = len(toks)
                st["w_total"] += nw
                st["c_total"] += len(content)
                st["n_unclear"] += low.count("[unclear]")
                brs = BRACKET_RE.findall(low)
                st["n_bracket"] += len(brs)
                brackets.update(brs)
                tok_set.update(TOKEN_RE.findall(low))
                if role == "tutor":
                    st["n_tutor"] += 1
                    st["w_tutor"] += nw
                    st["tutor_q"] += content.count("?")
                    if not greet_name and st["n_utt"] <= 30:
                        m = GREET_RE.search(content)
                        if m and m.group(1) not in NOT_NAMES:
                            greet_name = m.group(1)
                elif role == "student":
                    st["n_student"] += 1
                    st["w_student"] += nw
                    st["student_q"] += content.count("?")
                elif role == "background":
                    st["n_background"] += 1
                else:
                    st["n_other_role"] += 1
                ts = parse_timestamp(row.get("timestamp") or "")
                if ts is None:
                    st["n_ts_bad"] += 1
                else:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
                    if prev_ts is not None and ts < prev_ts:
                        st["ts_nonmono"] += 1
                    prev_ts = ts
    except Exception:
        st["parse_error"] = 1
    st["dur_s"] = (last_ts - first_ts) if (first_ts is not None and last_ts is not None) else np.nan
    st["content_hash"] = md5.hexdigest()
    st["greet_name"] = greet_name
    # lexical overlap between each linked learning objective and the transcript
    overlaps = []
    for lo_id, words in _WORKER_OBJ.get(session_id, []):
        if words:
            hit = sum(1 for w in words if w in tok_set)
            overlaps.append((session_id, lo_id, hit / len(words)))
    return st, brackets, roles, overlaps


_TEXT_TDIR: Path | None = None

def _init_text_worker(tdir_str: str):
    global _TEXT_TDIR
    _TEXT_TDIR = Path(tdir_str)


def read_session_text(sid: str) -> str:
    """Concatenated tutor+student utterances of one session (for the text baseline)."""
    import csv
    parts = []
    try:
        with open(_TEXT_TDIR / f"{sid}.csv", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if (row.get("role") or "") in ("tutor", "student"):
                    parts.append(row.get("content") or "")
    except FileNotFoundError:
        pass
    return " ".join(parts)


def run_transcript_pass(tdir: Path, features: pd.DataFrame, out: Path, workers: int, force: bool):
    cache = out / "transcript_stats.csv"
    meta_cache = out / "transcript_pass_meta.json"
    if cache.exists() and meta_cache.exists() and not force:
        print(f"[transcripts] using cache {cache}")
        stats = pd.read_csv(cache)
        meta = json.loads(meta_cache.read_text())
        return stats, Counter(meta["brackets"]), Counter(meta["roles"]), meta["parse_errors"]

    obj_map: dict[str, list[tuple[str, frozenset]]] = {}
    for sid, lo_id, lo_text in features[["session_id", "learning_objective_id",
                                         "learning_objective"]].drop_duplicates().itertuples(index=False):
        words = frozenset(w for w in TOKEN_RE.findall(str(lo_text).lower())
                          if w not in STOPWORDS and len(w) > 2)
        obj_map.setdefault(sid, []).append((lo_id, words))

    files = sorted(tdir.glob("*.csv"))
    print(f"[transcripts] parsing {len(files)} files with {workers} workers ...")
    t0 = time.time()
    rows, br_all, role_all, ov_all = [], Counter(), Counter(), []
    with Pool(workers, initializer=_init_worker, initargs=(obj_map,)) as pool:
        for st, brs, roles, ovs in pool.imap_unordered(parse_transcript, map(str, files), chunksize=64):
            rows.append(st)
            br_all.update(brs)
            role_all.update(roles)
            ov_all.extend(ovs)
    print(f"[transcripts] done in {time.time()-t0:.1f}s")
    stats = pd.DataFrame(rows)
    ov = pd.DataFrame(ov_all, columns=["session_id", "learning_objective_id", "lo_overlap"])
    stats = stats.merge(ov.groupby("session_id")["lo_overlap"].mean().rename("lo_overlap_mean"),
                        on="session_id", how="left")
    parse_errors = int(stats["parse_error"].sum())
    stats.to_csv(cache, index=False)
    ov.to_csv(out / "objective_overlap.csv", index=False)
    meta_cache.write_text(json.dumps({"brackets": dict(br_all.most_common(50)),
                                      "roles": dict(role_all), "parse_errors": parse_errors}))
    return stats, br_all, role_all, parse_errors


# ------------------------------------------------------------------ helpers ---
def logloss(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def binned_rate(ax, x, y, bins=10, logx=False, xlabel="", prior=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    qs = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return
    idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
    cx, cy, ce = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        n = m.sum()
        if n < 20:
            continue
        p_ = y[m].mean()
        cx.append(np.median(x[m]))
        cy.append(p_)
        ce.append(1.96 * np.sqrt(p_ * (1 - p_) / n))
    if prior is not None:
        ax.axhline(prior, color=P["baseline"], lw=1, ls="--", zorder=1)
    ax.errorbar(cx, cy, yerr=ce, color=P["blue"], lw=2, marker="o", ms=4,
                capsize=2, ecolor=P["baseline"], zorder=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", default=None)
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    data = Path(args.data_dir).expanduser()
    tdir = Path(args.transcripts_dir).expanduser() if args.transcripts_dir else data / "train_transcripts"
    out = Path(args.out).expanduser()
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    rng = np.random.default_rng(RNG_SEED)
    t_start = time.time()
    R: list[str] = []          # report lines
    R.append("# Trace the Ace — EDA report\n")
    R.append(f"Generated by `src/ace/eda/run_eda.py` (seed {RNG_SEED}). "
             f"All numbers below are reproducible from the raw data.\n")

    # ============================================ 1. load + integrity checks
    feats = pd.read_csv(data / "train_features.csv")
    labels = pd.read_csv(data / "train_labels.csv")
    # Test-side submission formats only feed the integrity table; training needs none of them.
    sub_test_f, sub_smoke_f = data / "submission_format_test.csv", data / "submission_format_smoke.csv"
    sub_test = pd.read_csv(sub_test_f) if sub_test_f.exists() else None
    sub_smoke = pd.read_csv(sub_smoke_f) if sub_smoke_f.exists() else None
    feats["learning_objective"] = feats["learning_objective"].str.strip()

    df = feats.merge(labels, on="response_id", how="left", validate="1:1")
    y = df["is_correct"].to_numpy()
    prior = float(y.mean())

    R.append("## 1. Data integrity\n")
    integ = []
    integ.append(("train rows", len(df)))
    integ.append(("duplicate response_id", int(feats["response_id"].duplicated().sum())))
    integ.append(("rows with missing label", int(df["is_correct"].isna().sum())))
    integ.append(("nulls in features", int(feats.isna().sum().sum())))
    integ.append(("unique session_id", feats["session_id"].nunique()))
    integ.append(("unique learning_objective_id", feats["learning_objective_id"].nunique()))
    integ.append(("unique learning_objective text", feats["learning_objective"].nunique()))
    lo_txt_per_id = feats.groupby("learning_objective_id")["learning_objective"].nunique()
    lo_id_per_txt = feats.groupby("learning_objective")["learning_objective_id"].nunique()
    integ.append(("objective ids with >1 text", int((lo_txt_per_id > 1).sum())))
    integ.append(("objective texts with >1 id", int((lo_id_per_txt > 1).sum())))
    dup_pair = df.groupby(["session_id", "learning_objective_id"]).size()
    n_dup_pair = int((dup_pair > 1).sum())
    integ.append(("(session, objective) pairs with >1 response", n_dup_pair))
    if n_dup_pair:
        dp = df.merge(dup_pair[dup_pair > 1].rename("n").reset_index(),
                      on=["session_id", "learning_objective_id"])
        agree = dp.groupby(["session_id", "learning_objective_id"])["is_correct"].nunique().eq(1).mean()
        integ.append(("... of which label-consistent", f"{agree:.1%}"))
    if sub_test is not None:
        integ.append(("test rows (submission_format_test)", len(sub_test)))
        integ.append(("test response_id overlap with train", int(sub_test["response_id"].isin(feats["response_id"]).sum())))
    if sub_smoke is not None:
        integ.append(("smoke rows", len(sub_smoke)))
    if sub_test is not None and sub_smoke is not None:
        integ.append(("smoke response_id ⊂ test response_id", bool(sub_smoke["response_id"].isin(sub_test["response_id"]).all())))
    R.append(md_table(["check", "value"], integ) + "\n")

    # ==================================================== 2. transcript pass
    stats, brackets, roles, parse_errors = run_transcript_pass(tdir, feats, out, args.workers, args.force)
    stats = stats.set_index("session_id")
    sess_feat = set(feats["session_id"])
    missing_files = sorted(sess_feat - set(stats.index))
    extra_files = sorted(set(stats.index) - sess_feat)

    st = stats.copy()
    st["student_share_w"] = st["w_student"] / st["w_total"].clip(lower=1)
    st["w_per_utt_student"] = st["w_student"] / st["n_student"].clip(lower=1)
    st["w_per_utt_tutor"] = st["w_tutor"] / st["n_tutor"].clip(lower=1)
    st["unclear_share"] = st["n_unclear"] / st["n_utt"].clip(lower=1)
    st["utt_gap_share"] = 1 - st["n_utt"] / (st["max_utt_id"] + 1).clip(lower=1)
    st["est_tokens"] = st["w_total"] * 1.35

    R.append("## 2. Transcript corpus overview\n")
    R.append(f"- Files parsed: **{len(stats)}** (parse errors: {parse_errors}; "
             f"missing for train sessions: {len(missing_files)}; extra files: {len(extra_files)})")
    R.append(f"- Roles found: `{dict(roles)}`")
    R.append(f"- Total words: **{st['w_total'].sum()/1e6:.1f}M** "
             f"(student {st['w_student'].sum()/1e6:.1f}M, tutor {st['w_tutor'].sum()/1e6:.1f}M)")
    q = st[["n_utt", "w_total", "w_student", "w_tutor", "dur_s", "est_tokens"]].describe(
        percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    R.append("\n**Per-transcript distributions**\n")
    R.append(md_table(["stat"] + list(q.columns),
                      [[i] + [f"{v:,.0f}" for v in q.loc[i]] for i in q.index]) + "\n")
    ctx = [(w, f"{(st['est_tokens'] <= w).mean():.1%}") for w in (4096, 8192, 16384, 32768, 131072)]
    R.append("**LLM context feasibility** (estimated tokens = words × 1.35): share of transcripts fitting each window\n")
    R.append(md_table(["window", "fits"], ctx) + "\n")
    R.append("**Top bracketed transcript artifacts** (ASR noise markers)\n")
    R.append(md_table(["token", "occurrences"], [(f"`[{k}]`", f"{v:,}") for k, v in brackets.most_common(12)]) + "\n")

    # ------------------------------------------------------------- bad data
    R.append("## 3. Bad / suspicious data\n")
    bad = []
    bad.append(("transcripts with 0 utterances", int((st["n_utt"] == 0).sum())))
    bad.append(("transcripts with 0 student utterances", int((st["n_student"] == 0).sum())))
    bad.append(("transcripts with 0 tutor utterances", int((st["n_tutor"] == 0).sum())))
    bad.append(("transcripts with < 50 total words", int((st["w_total"] < 50).sum())))
    bad.append(("transcripts with < 20 student words", int((st["w_student"] < 20).sum())))
    bad.append(("duration missing / all-bad timestamps", int(st["dur_s"].isna().sum())))
    bad.append(("duration == 0s", int((st["dur_s"] == 0).sum())))
    bad.append(("non-monotonic timestamp steps (files affected)", int((st["ts_nonmono"] > 0).sum())))
    bad.append(("unclear share > 20% of utterances", int((st["unclear_share"] > 0.2).sum())))
    bad.append(("empty-content rows (total)", int(st["n_empty"].sum())))
    hash_dup = st.groupby("content_hash").size()
    ndup = int((hash_dup > 1).sum())
    bad.append(("duplicate transcript content (hash groups >1)", ndup))
    bad.append(("rows in duplicated-content sessions", int(hash_dup[hash_dup > 1].sum()) if ndup else 0))
    R.append(md_table(["issue", "count"], bad) + "\n")
    if ndup:
        dup_sessions = st[st["content_hash"].isin(hash_dup[hash_dup > 1].index)]
        dup_sessions.reset_index()[["session_id", "content_hash", "n_utt", "w_total"]] \
            .sort_values("content_hash").to_csv(out / "duplicate_transcripts.csv", index=False)
        R.append(f"Duplicated-content sessions written to `duplicate_transcripts.csv`. "
                 f"**If duplicates cross CV folds, group by content hash, not just session_id.**\n")

    names = st.loc[st["greet_name"] != "", "greet_name"]
    R.append(f"**Student first names** greeted by the tutor were detected in "
             f"{len(names):,} sessions ({len(names)/len(st):.0%}) — {names.nunique():,} distinct names; "
             f"top: {', '.join(f'{k} ({v})' for k, v in names.value_counts().head(8).items())}. "
             f"Names recur across sessions → a latent student-level grouping may exist "
             f"(unusable as ID, but a caution for over-optimistic CV and a hint that "
             f"student-level signal exists).\n")

    # =========================================================== 3. labels
    R.append("## 4. Labels\n")
    R.append(f"- Target rate (prior): **{prior:.4f}** — constant-prior log loss "
             f"**{logloss(y, np.full_like(y, prior)):.4f}**, 0.5-constant {logloss(y, np.full_like(y, .5)):.4f}\n")

    rp = df.groupby("session_id").size().rename("n_resp")
    df = df.merge(rp, on="session_id")
    by_n = df.groupby("n_resp")["is_correct"].agg(["size", "mean"])
    R.append("**Target rate by responses-per-session** (Case #1 = 1 response)\n")
    R.append(md_table(["responses in session", "rows", "target rate"],
                      [(i, f"{r['size']:,.0f}", f"{r['mean']:.3f}") for i, r in by_n.iterrows()]) + "\n")

    # within-session label agreement vs. permutation null
    multi = df[df["n_resp"] > 1]
    def all_same_rate(vals, sess):
        g = pd.Series(vals).groupby(sess).nunique()
        return float((g == 1).mean())
    obs = all_same_rate(multi["is_correct"].values, multi["session_id"].values)
    perm = []
    mv = multi["is_correct"].values.copy()
    for _ in range(200):
        rng.shuffle(mv)
        perm.append(all_same_rate(mv, multi["session_id"].values))
    perm = np.array(perm)
    R.append(f"**Within-session label agreement** (multi-response sessions, n={multi['session_id'].nunique():,}): "
             f"all labels identical in **{obs:.1%}** of sessions vs **{perm.mean():.1%} ± {perm.std():.1%}** "
             f"under random shuffling → a strong session-level effect. "
             f"Session-level features carry real signal, and per-objective differentiation "
             f"within a session is the harder, rarer event.\n")

    # per-size agreement figure
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    sizes = [2, 3, 4]
    obs_s, per_s = [], []
    for k in sizes:
        sub = df[df["n_resp"] == k] if k < 4 else df[df["n_resp"] >= 4]
        o = all_same_rate(sub["is_correct"].values, sub["session_id"].values)
        pv = []
        sv = sub["is_correct"].values.copy()
        for _ in range(100):
            rng.shuffle(sv)
            pv.append(all_same_rate(sv, sub["session_id"].values))
        obs_s.append(o); per_s.append(np.mean(pv))
    xpos = np.arange(len(sizes))
    ax.bar(xpos - 0.19, obs_s, 0.34, color=P["blue"], label="observed")
    ax.bar(xpos + 0.19, per_s, 0.34, color=P["aqua"], label="shuffled labels")
    for xp, v in zip(xpos - 0.19, obs_s):
        ax.text(xp, v + .01, f"{v:.0%}", ha="center", fontsize=8, color=P["ink2"])
    for xp, v in zip(xpos + 0.19, per_s):
        ax.text(xp, v + .01, f"{v:.0%}", ha="center", fontsize=8, color=P["ink2"])
    ax.set_xticks(xpos, ["2 responses", "3 responses", "4+ responses"])
    ax.set_ylabel("sessions with all labels identical")
    ax.set_title("Within-session label agreement far exceeds chance")
    ax.legend(); ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f07_within_session_agreement.png"); plt.close(fig)

    # ------------------------------------------------- learning objectives
    R.append("## 5. Learning objectives\n")
    lo = df.groupby(["learning_objective_id", "learning_objective"])["is_correct"].agg(["size", "mean"]).reset_index()
    lo = lo.sort_values("size", ascending=False)
    top_share = lo["size"].head(20).sum() / len(df)
    R.append(f"- {len(lo)} unique objectives; top-20 cover {top_share:.1%} of rows; "
             f"{int((lo['size'] < 10).sum())} objectives have <10 rows.")
    # smoothed rates for ranking
    alpha = 30
    lo["rate_smoothed"] = (lo["size"] * lo["mean"] + alpha * prior) / (lo["size"] + alpha)
    show = lambda d: [(t[:70], f"{s:,}", f"{m:.3f}") for t, s, m in
                      d[["learning_objective", "size", "mean"]].itertuples(index=False)]
    R.append("\n**Easiest objectives** (smoothed, ≥100 rows)\n")
    big = lo[lo["size"] >= 100]
    R.append(md_table(["objective", "rows", "rate"], show(big.nlargest(10, "rate_smoothed"))) + "\n")
    R.append("**Hardest objectives** (smoothed, ≥100 rows)\n")
    R.append(md_table(["objective", "rows", "rate"], show(big.nsmallest(10, "rate_smoothed"))) + "\n")

    # objective-only model (OOF, grouped): how much is the objective id worth?
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    folds = np.full(len(df), -1)
    for f_, (_, va) in enumerate(sgkf.split(df, df["is_correct"], groups=df["session_id"])):
        folds[va] = f_
    df["fold"] = folds
    oof_lo = np.full(len(df), np.nan)
    for f_ in range(5):
        tr, va = df["fold"] != f_, df["fold"] == f_
        r = df[tr].groupby("learning_objective_id")["is_correct"].agg(["sum", "size"])
        rate = ((r["sum"] + alpha * df.loc[tr, "is_correct"].mean()) / (r["size"] + alpha))
        oof_lo[va] = df.loc[va, "learning_objective_id"].map(rate).fillna(prior).values
    from sklearn.metrics import roc_auc_score
    R.append(f"**Objective-only model** (out-of-fold smoothed target rate per `learning_objective_id`): "
             f"log loss **{logloss(y, oof_lo):.4f}**, AUC **{roc_auc_score(y, oof_lo):.4f}** "
             f"(vs prior baseline {logloss(y, np.full_like(y, prior)):.4f}).\n")

    # frequency + coverage figure
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(np.arange(1, len(lo) + 1), lo["size"].values, color=P["blue"], lw=2)
    axes[0].set_yscale("log"); axes[0].set_xlabel("objective rank"); axes[0].set_ylabel("rows (log)")
    axes[0].set_title("Objective frequency is long-tailed")
    axes[1].plot(np.arange(1, len(lo) + 1), lo["size"].cumsum().values / len(df), color=P["blue"], lw=2)
    axes[1].set_xlabel("objective rank"); axes[1].set_ylabel("cumulative share of rows")
    axes[1].set_title("Cumulative coverage")
    axes[1].axhline(0.8, color=P["baseline"], ls="--", lw=1)
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f04_objective_frequency.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.scatter(lo["size"], lo["mean"], s=14, color=P["blue"], alpha=.55, linewidths=0)
    ax.axhline(prior, color=P["baseline"], ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("rows per objective (log)"); ax.set_ylabel("target rate")
    ax.set_title("Objective difficulty varies widely — real signal, noisy tail")
    ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f05_objective_rate_vs_freq.png"); plt.close(fig)

    # ------------------------------------------------------------- topics
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    texts = lo["learning_objective"].tolist()
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X_lo = vec.fit_transform(texts)
    k = 14
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(X_lo)
    terms = np.array(vec.get_feature_names_out())
    cl_label = {c: ", ".join(terms[np.argsort(km.cluster_centers_[c])[::-1][:3]]) for c in range(k)}
    lo["topic"] = km.labels_
    lo["topic_label"] = lo["topic"].map(cl_label)
    df = df.merge(lo[["learning_objective_id", "topic", "topic_label"]], on="learning_objective_id", how="left")
    top = df.groupby("topic_label")["is_correct"].agg(["size", "mean"]).sort_values("mean")
    lo.to_csv(out / "objective_topics.csv", index=False)
    R.append("## 6. Topics (KMeans on objective TF-IDF, k=14)\n")
    R.append(md_table(["topic (top terms)", "rows", "target rate"],
                      [(i, f"{r['size']:,.0f}", f"{r['mean']:.3f}") for i, r in top.iterrows()]) + "\n")
    R.append("Full objective→topic mapping in `objective_topics.csv`.\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ypos = np.arange(len(top))
    ax.barh(ypos, top["mean"].values, color=P["blue"], height=0.62)
    ax.axvline(prior, color=P["baseline"], ls="--", lw=1)
    ax.set_yticks(ypos, [f"{i}  (n={r['size']:,.0f})" for i, r in top.iterrows()], fontsize=8)
    ax.set_xlabel("target rate"); ax.set_xlim(0, 1)
    ax.set_title(f"Target rate by objective topic (dashed = prior {prior:.2f})")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f06_target_by_topic.png"); plt.close(fig)

    # ================================================ 4. transcripts vs y
    dt = df.merge(st.reset_index(), on="session_id", how="left")
    ov = pd.read_csv(out / "objective_overlap.csv")
    dt = dt.merge(ov, on=["session_id", "learning_objective_id"], how="left")

    R.append("## 7. Transcript signals vs target\n")
    feats_num = [("w_total", "total words", True), ("dur_s", "duration (s)", False),
                 ("student_share_w", "student word share", False),
                 ("w_per_utt_student", "words / student utterance", True),
                 ("unclear_share", "[unclear] share", False),
                 ("lo_overlap", "objective-transcript lexical overlap", False)]
    corr_rows = []
    for c, lbl, _ in feats_num + [("tutor_q", "tutor '?' count", True), ("n_utt", "utterances", True)]:
        v = dt[c].astype(float)
        ok = v.notna()
        corr = np.corrcoef(v[ok], dt.loc[ok, "is_correct"])[0, 1]
        corr_rows.append((lbl, f"{corr:+.3f}"))
    R.append("**Point-biserial correlation with `is_correct`**\n")
    R.append(md_table(["transcript feature", "corr"], corr_rows) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6))
    for ax, (c, lbl, logx) in zip(axes.ravel(), feats_num):
        binned_rate(ax, dt[c], dt["is_correct"], bins=10, logx=logx, xlabel=lbl, prior=prior)
    axes[0][0].set_ylabel("target rate"); axes[1][0].set_ylabel("target rate")
    fig.suptitle("Target rate across transcript-feature deciles (dashed = prior)",
                 fontsize=11, fontweight="bold", color=P["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(figdir / "f03_target_by_transcript_bins.png"); plt.close(fig)

    # distribution figures
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(st["w_total"].clip(1), bins=np.geomspace(1, st["w_total"].max() + 1, 60),
                 color=P["blue"], edgecolor=P["surface"], lw=.4)
    axes[0].set_xscale("log"); axes[0].set_xlabel("words per transcript (log)"); axes[0].set_ylabel("sessions")
    axes[0].set_title("Transcript length")
    axes[1].hist(st["dur_s"].dropna() / 60, bins=60, color=P["blue"], edgecolor=P["surface"], lw=.4)
    axes[1].set_xlabel("duration (min)"); axes[1].set_title("Session duration")
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f01_length_duration.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(st["w_per_utt_student"].clip(0, 40), bins=60, color=P["blue"],
                 edgecolor=P["surface"], lw=.4)
    axes[0].set_xlabel("words per student utterance (clipped at 40)")
    axes[0].set_ylabel("sessions"); axes[0].set_title("Student verbosity")
    axes[1].hist(st["student_share_w"], bins=50, color=P["blue"], edgecolor=P["surface"], lw=.4)
    axes[1].set_xlabel("student share of words"); axes[1].set_title("Who talks?")
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f02_style.png"); plt.close(fig)

    # ======================================================== 5. CV design
    R.append("## 8. CV design\n")
    fold_rows = []
    for f_ in range(5):
        va = df[df["fold"] == f_]
        tr = df[df["fold"] != f_]
        unseen = (~va["learning_objective_id"].isin(set(tr["learning_objective_id"]))).mean()
        fold_rows.append((f_, f"{len(va):,}", f"{va['session_id'].nunique():,}",
                          f"{va['is_correct'].mean():.4f}",
                          va["learning_objective_id"].nunique(), f"{unseen:.2%}"))
    R.append(f"Recommended protocol: **StratifiedGroupKFold(5, shuffle, seed={RNG_SEED})** grouped by "
             f"`session_id`, stratified on the label. Fixed assignment saved to `cv_folds.csv` — "
             f"reuse it for every experiment.\n")
    R.append(md_table(["fold", "rows", "sessions", "target rate", "objectives", "rows w/ unseen objective"],
                      fold_rows) + "\n")
    df[["response_id", "session_id", "fold"]].to_csv(out / "cv_folds.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    fr = df.groupby("fold")["is_correct"].agg(["size", "mean"])
    axes[0].bar(fr.index, fr["size"], color=P["blue"], width=.6)
    axes[0].set_xlabel("fold"); axes[0].set_ylabel("rows"); axes[0].set_title("Fold sizes")
    axes[1].plot(fr.index, fr["mean"], color=P["blue"], marker="o", lw=2)
    axes[1].axhline(prior, color=P["baseline"], ls="--", lw=1)
    axes[1].set_xlabel("fold"); axes[1].set_ylabel("target rate"); axes[1].set_title("Fold target rates")
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "f08_cv_folds.png"); plt.close(fig)

    # --------------------------------------- leakage demo + text baseline
    if not args.skip_baseline:
        R.append("## 9. Leakage demonstration + first text baseline\n")
        print("[baseline] loading transcript text ...")
        sids = df["session_id"].drop_duplicates().tolist()
        with Pool(args.workers, initializer=_init_text_worker, initargs=(str(tdir),)) as pool:
            texts_by_sid = dict(zip(sids, pool.map(read_session_text, sids, chunksize=64)))

        print("[baseline] TF-IDF ...")
        tvec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=300_000,
                               sublinear_tf=True)
        Xt_sess = tvec.fit_transform([texts_by_sid[s] for s in sids])
        sid_ix = {s: i for i, s in enumerate(sids)}
        Xt = Xt_sess[[sid_ix[s] for s in df["session_id"]]]
        ovec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        Xo = ovec.fit_transform(df["learning_objective"])
        from scipy.sparse import hstack
        X = hstack([Xt, Xo * 2.0]).tocsr()

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        def run_cv(fold_arr, name):
            oof = np.full(len(df), np.nan)
            for f_ in sorted(np.unique(fold_arr)):
                tr, va = fold_arr != f_, fold_arr == f_
                clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
                clf.fit(X[tr], y[tr])
                oof[va] = clf.predict_proba(X[va])[:, 1]
                print(f"  [{name}] fold {f_}: ll={logloss(y[va], oof[va]):.4f}")
            return oof

        print("[baseline] grouped CV ...")
        oof_g = run_cv(df["fold"].values, "group")
        print("[baseline] naive random CV (leaky) ...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
        f_rand = np.full(len(df), -1)
        for f_, (_, va) in enumerate(skf.split(df, y)):
            f_rand[va] = f_
        oof_r = run_cv(f_rand, "random")

        ll_g, ll_r = logloss(y, oof_g), logloss(y, oof_r)
        auc_g, auc_r = roc_auc_score(y, oof_g), roc_auc_score(y, oof_r)
        R.append("TF-IDF (transcript + objective, word 1-2 grams) + logistic regression:\n")
        R.append(md_table(["CV scheme", "log loss", "AUC"],
                          [("StratifiedGroupKFold by session (correct)", f"{ll_g:.4f}", f"{auc_g:.4f}"),
                           ("StratifiedKFold random rows (leaky)", f"{ll_r:.4f}", f"{auc_r:.4f}"),
                           ("prior baseline", f"{logloss(y, np.full_like(y, prior)):.4f}", "0.5000"),
                           ("objective-only model", f"{logloss(y, oof_lo):.4f}", f"{roc_auc_score(y, oof_lo):.4f}")]) + "\n")
        R.append(f"The random-row split looks **{(ll_g - ll_r):.4f} log-loss better** than the honest "
                 f"grouped split purely because rows sharing a transcript land in both train and "
                 f"validation. Never trust a random split in this competition.\n")
        pd.DataFrame({"response_id": df["response_id"], "oof_group": oof_g,
                      "oof_random": oof_r, "oof_objective_only": oof_lo}).to_csv(
            out / "baseline_oof.csv", index=False)

        # reliability + leakage figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        bins = np.linspace(0, 1, 11)
        bi = np.clip(np.digitize(oof_g, bins) - 1, 0, 9)
        mp = [oof_g[bi == b].mean() for b in range(10) if (bi == b).sum() > 30]
        ma = [y[bi == b].mean() for b in range(10) if (bi == b).sum() > 30]
        axes[0].plot([0, 1], [0, 1], color=P["baseline"], ls="--", lw=1)
        axes[0].plot(mp, ma, color=P["blue"], marker="o", lw=2)
        axes[0].set_xlabel("mean predicted probability"); axes[0].set_ylabel("observed rate")
        axes[0].set_title("Reliability of grouped-CV OOF predictions")
        b = axes[1].bar(["grouped CV\n(honest)", "random CV\n(leaky)"], [ll_g, ll_r],
                        color=[P["blue"], P["aqua"]], width=.55)
        axes[1].bar_label(b, fmt="%.4f", fontsize=9, color=P["ink2"])
        axes[1].axhline(logloss(y, np.full_like(y, prior)), color=P["baseline"], ls="--", lw=1)
        axes[1].set_ylabel("log loss (lower = better)")
        axes[1].set_title("Leakage inflates random-split scores")
        for ax in axes:
            ax.grid(axis="y"); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(figdir / "f09_baseline_leakage.png"); plt.close(fig)

    # ---------------------------------------------------------------- misc
    R.append("## 10. Case #1 vs Case #2 and transcript size\n")
    dt["case"] = np.where(dt["n_resp"] == 1, "case1 (1 resp)", "case2 (2+ resp)")
    cs = dt.groupby("case").agg(rows=("is_correct", "size"), rate=("is_correct", "mean"),
                                words=("w_total", "median"), dur=("dur_s", "median"))
    R.append(md_table(["case", "rows", "target rate", "median words", "median dur (s)"],
                      [(i, f"{r['rows']:,.0f}", f"{r['rate']:.3f}", f"{r['words']:,.0f}",
                        f"{r['dur']:,.0f}") for i, r in cs.iterrows()]) + "\n")

    R.append("## Figures\n")
    for f_ in sorted(figdir.glob("*.png")):
        R.append(f"![{f_.stem}](figures/{f_.name})\n")
    R.append(f"\n_Total EDA runtime: {time.time()-t_start:.0f}s_\n")
    (out / "REPORT.md").write_text("\n".join(R), encoding="utf-8")
    print(f"\nDone. Report: {out/'REPORT.md'}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()