#!/usr/bin/env python
r"""Embedding pass (training stage 2) with BAAI/bge-base-en-v1.5: taxonomy, coverage, topics.

1. objective taxonomy: near-duplicate objectives, embedding topic clusters, and the kNN-backoff
   target encoding on the fixed CV folds;
2. objective-to-transcript coverage: max cosine between the objective and 200-word transcript
   windows, and its relation to the label;
3. UMAP datamaps of objectives and sessions.
Writes outputs/embeddings/ (REPORT.md, coverage.csv, objective_topics_emb.csv, figures) and caches
the embeddings in --cache-dir (segment embeddings ~800 MB fp16), so re-runs are fast.

    python src/ace/eda/run_embeddings.py --data-dir data --transcripts-dir data/train_transcripts \
        --eda-dir outputs/eda --out outputs/embeddings --cache-dir cache
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
SEG_WORDS = 200          # words per transcript window
SEG_STRIDE = 150         # window stride (25% overlap)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

P = {
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "baseline": "#c3c2b7",
    "blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
    "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834",
}
SERIES = [P["blue"], P["aqua"], P["yellow"], P["green"], P["violet"], P["red"], P["magenta"]]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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


# ------------------------------------------------------------------ file IO ---
_TDIR: Path | None = None

def _init_worker(tdir_str: str):
    global _TDIR
    _TDIR = Path(tdir_str)


def read_session_words(sid: str):
    """Tutor+student utterances of one session, as a word list."""
    import csv
    words: list[str] = []
    try:
        with open(_TDIR / f"{sid}.csv", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if (row.get("role") or "") in ("tutor", "student"):
                    words.extend((row.get("content") or "").split())
    except FileNotFoundError:
        pass
    return sid, words


def logloss(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--eda-dir", default="outputs/eda")
    ap.add_argument("--out", default="outputs/embeddings")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = Path(args.data_dir).expanduser()
    tdir = Path(args.transcripts_dir).expanduser()
    eda = Path(args.eda_dir).expanduser()
    out = Path(args.out).expanduser()
    cache = Path(args.cache_dir).expanduser()
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    plt = setup_mpl()
    t_start = time.time()

    feats = pd.read_csv(data / "train_features.csv")
    labels = pd.read_csv(data / "train_labels.csv")
    feats["learning_objective"] = feats["learning_objective"].str.strip()
    df = feats.merge(labels, on="response_id", validate="1:1")
    folds = pd.read_csv(eda / "cv_folds.csv")[["response_id", "fold"]]
    df = df.merge(folds, on="response_id", validate="1:1")
    y = df["is_correct"].to_numpy()
    prior = float(y.mean())

    lo = (df.groupby(["learning_objective_id", "learning_objective"])["is_correct"]
            .agg(["size", "mean"]).reset_index()
            .sort_values("learning_objective_id").reset_index(drop=True))
    lo_ix = {k: i for i, k in enumerate(lo["learning_objective_id"])}
    sids = sorted(df["session_id"].unique())

    R: list[str] = ["# Trace the Ace — embedding pass report\n",
                    f"Model: `{args.model}` (MIT license, open weights — competition-compatible). "
                    f"Windows: {SEG_WORDS} words, stride {SEG_STRIDE}. Seed {RNG_SEED}. "
                    f"Generated by `src/ace/eda/run_embeddings.py`.\n"]

    # ------------------------------------------------------------ embeddings
    emb_obj_f = cache / "emb_objectives.npy"        # plain, for taxonomy/kNN
    emb_query_f = cache / "emb_objectives_query.npy"  # bge query-prefixed, for coverage
    emb_seg_f = cache / "emb_segments_f16.npy"
    seg_meta_f = cache / "segments_meta.json"

    if not (emb_obj_f.exists() and emb_query_f.exists() and emb_seg_f.exists()
            and seg_meta_f.exists()) or args.force:
        from sentence_transformers import SentenceTransformer
        print(f"[embed] loading {args.model} ...")
        model = SentenceTransformer(args.model, device="cuda")

        print("[embed] objectives ...")
        texts = lo["learning_objective"].tolist()
        emb_obj = model.encode(texts, batch_size=64, normalize_embeddings=True,
                               convert_to_numpy=True)
        emb_query = model.encode([BGE_QUERY_PREFIX + t for t in texts], batch_size=64,
                                 normalize_embeddings=True, convert_to_numpy=True)
        np.save(emb_obj_f, emb_obj.astype(np.float32))
        np.save(emb_query_f, emb_query.astype(np.float32))

        print(f"[embed] reading {len(sids)} transcripts ...")
        with Pool(args.workers, initializer=_init_worker, initargs=(str(tdir),)) as pool:
            words_by_sid = dict(pool.map(read_session_words, sids, chunksize=64))

        print("[embed] segmenting ...")
        segments, bounds = [], {}
        for sid in sids:
            w = words_by_sid[sid]
            start = len(segments)
            if not w:
                bounds[sid] = (start, start)
                continue
            for i in range(0, max(1, len(w) - SEG_STRIDE), SEG_STRIDE):
                seg = w[i:i + SEG_WORDS]
                if len(seg) >= 30 or i == 0:
                    segments.append(" ".join(seg))
            bounds[sid] = (start, len(segments))
        print(f"[embed] {len(segments):,} segments -> encoding on GPU ...")
        emb_seg = model.encode(segments, batch_size=args.batch_size,
                               normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=True)
        np.save(emb_seg_f, emb_seg.astype(np.float16))
        seg_meta_f.write_text(json.dumps({s: list(b) for s, b in bounds.items()}))
        del model
    else:
        print("[embed] using cached embeddings")

    emb_obj = np.load(emb_obj_f)
    emb_query = np.load(emb_query_f)
    emb_seg = np.load(emb_seg_f).astype(np.float32)
    bounds = {k: tuple(v) for k, v in json.loads(seg_meta_f.read_text()).items()}
    R.append(f"Corpus: {len(lo)} objectives, {len(sids):,} sessions, "
             f"{emb_seg.shape[0]:,} transcript windows embedded ({emb_seg.shape[1]}d).\n")

    # ========================================== 1. taxonomy audit (objectives)
    R.append("## 1. Objective taxonomy audit\n")
    S = emb_obj @ emb_obj.T
    np.fill_diagonal(S, -1)
    iu = np.triu_indices(len(lo), k=1)
    sims = S[iu]
    order = np.argsort(sims)[::-1]
    R.append(f"Pairwise objective similarity: median {np.median(sims):.3f}, "
             f"p95 {np.quantile(sims, .95):.3f}; **{int((sims > 0.92).sum())} pairs > 0.92** "
             f"(near-duplicates) and {int((sims > 0.85).sum())} pairs > 0.85 (siblings).\n")
    rows = []
    for k in order[:15]:
        i, j = iu[0][k], iu[1][k]
        rows.append((f"{sims[k]:.3f}",
                     f"{lo.loc[i, 'learning_objective'][:58]} (n={lo.loc[i, 'size']}, r={lo.loc[i, 'mean']:.2f})",
                     f"{lo.loc[j, 'learning_objective'][:58]} (n={lo.loc[j, 'size']}, r={lo.loc[j, 'mean']:.2f})"))
    R.append("**Top near-duplicate pairs**\n")
    R.append(md_table(["cos", "objective A", "objective B"], rows) + "\n")

    # embedding topic clusters vs TF-IDF KMeans topics
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, roc_auc_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    k_topics = 14
    agg = AgglomerativeClustering(n_clusters=k_topics, metric="cosine", linkage="average")
    lo["topic_emb"] = agg.fit_predict(emb_obj)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    Xtf = vec.fit_transform(lo["learning_objective"])
    terms = np.array(vec.get_feature_names_out())
    lab = {}
    for c in range(k_topics):
        m = (lo["topic_emb"] == c).to_numpy()
        lab[c] = ", ".join(terms[np.asarray(Xtf[m].mean(axis=0)).ravel().argsort()[::-1][:3]])
    lo["topic_emb_label"] = lo["topic_emb"].map(lab)
    old = pd.read_csv(eda / "objective_topics.csv")[["learning_objective_id", "topic"]]
    both = lo.merge(old, on="learning_objective_id")
    ari = adjusted_rand_score(both["topic"], both["topic_emb"])
    d2 = df.merge(lo[["learning_objective_id", "topic_emb", "topic_emb_label"]],
                  on="learning_objective_id")
    tt = d2.groupby("topic_emb_label")["is_correct"].agg(["size", "mean"]).sort_values("mean")
    R.append(f"**Embedding topics (Agglomerative cosine, k={k_topics})** — agreement with the "
             f"TF-IDF KMeans topics: adjusted Rand index **{ari:.2f}** "
             f"(1 = identical, 0 = random).\n")
    R.append(md_table(["topic (top terms)", "rows", "target rate"],
                      [(i, f"{r['size']:,.0f}", f"{r['mean']:.3f}") for i, r in tt.iterrows()]) + "\n")
    lo.to_csv(out / "objective_topics_emb.csv", index=False)

    # ------------------------------------------- kNN-backoff target encoding
    R.append("### kNN-backoff target encoding (rare-objective smoothing)\n")
    knn_k = 10
    nn_idx = np.argsort(-S, axis=1)[:, :knn_k]
    nn_sim = np.take_along_axis(S, nn_idx, axis=1).clip(min=0)
    lo_of_row = df["learning_objective_id"].map(lo_ix).to_numpy()
    fold_arr = df["fold"].to_numpy()

    def oof_te(a_knn: float, a_prior: float):
        oof = np.full(len(df), np.nan)
        for f in range(5):
            tr = fold_arr != f
            p_tr = y[tr].mean()
            cnt = np.bincount(lo_of_row[tr], minlength=len(lo)).astype(float)
            s = np.bincount(lo_of_row[tr], weights=y[tr], minlength=len(lo))
            rate = np.divide(s, cnt, out=np.full(len(lo), p_tr), where=cnt > 0)
            w = nn_sim * cnt[nn_idx]
            r_knn = (w * rate[nn_idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
            te = (s + a_knn * r_knn + a_prior * p_tr) / (cnt + a_knn + a_prior)
            oof[fold_arr == f] = te[lo_of_row[fold_arr == f]]
        return oof

    grid = []
    for a_knn, a_prior in [(0, 30), (10, 10), (20, 10), (40, 10), (20, 5)]:
        o = oof_te(a_knn, a_prior)
        grid.append((a_knn, a_prior, logloss(y, o), roc_auc_score(y, o), o))
    base = min(g for g in grid if g[0] == 0)
    best = min(grid, key=lambda g: g[2])
    R.append(md_table(["knn weight", "prior weight", "OOF log loss", "AUC"],
                      [(g[0], g[1], f"{g[2]:.4f}", f"{g[3]:.4f}") for g in grid]) + "\n")
    R.append(f"Borrowing strength from the {knn_k} nearest objectives improves the "
             f"objective-only model from **{base[2]:.4f}** (plain smoothing) to "
             f"**{best[2]:.4f}** log loss. The gain concentrates in rare objectives.\n")
    # rare-slice detail for best vs plain
    cnt_all = np.bincount(lo_of_row, minlength=len(lo))
    rare = cnt_all[lo_of_row] < 30
    R.append(md_table(["slice", "rows", "plain TE", "kNN-backoff TE"],
                      [("objective <30 train rows", f"{int(rare.sum()):,}",
                        f"{logloss(y[rare], base[4][rare]):.4f}", f"{logloss(y[rare], best[4][rare]):.4f}"),
                       ("objective ≥30 train rows", f"{int((~rare).sum()):,}",
                        f"{logloss(y[~rare], base[4][~rare]):.4f}", f"{logloss(y[~rare], best[4][~rare]):.4f}")]) + "\n")

    # =========================================== 2. coverage audit (retrieval)
    R.append("## 2. Objective→transcript coverage audit\n")
    cov_max = np.full(len(df), np.nan)
    cov_top3 = np.full(len(df), np.nan)
    by_sid = df.groupby("session_id").indices
    for sid, rows_ix in by_sid.items():
        a, b = bounds.get(sid, (0, 0))
        if b <= a:
            continue
        segs = emb_seg[a:b]
        for ix in rows_ix:
            sim = segs @ emb_query[lo_of_row[ix]]
            cov_max[ix] = sim.max()
            cov_top3[ix] = np.sort(sim)[-3:].mean()
    df["coverage_max"] = cov_max
    df["coverage_top3"] = cov_top3
    df[["response_id", "session_id", "learning_objective_id",
        "coverage_max", "coverage_top3"]].to_csv(out / "coverage.csv", index=False)

    c = df["coverage_max"]
    corr = np.corrcoef(c, y)[0, 1]
    R.append(f"`coverage_max` = max cosine between the (query-prefixed) objective and the "
             f"session's windows. Distribution: p5 {c.quantile(.05):.3f}, median {c.median():.3f}, "
             f"p95 {c.quantile(.95):.3f}. Point-biserial correlation with `is_correct`: "
             f"**{corr:+.3f}** (lexical overlap was +0.025).\n")

    dec = pd.qcut(c, 10, labels=False, duplicates="drop")
    dd = df.groupby(dec)["is_correct"].agg(["size", "mean"])
    # where is the text baseline blind?
    oof = pd.read_csv(eda / "baseline_oof.csv")[["response_id", "oof_group"]]
    df2 = df.merge(oof, on="response_id", validate="1:1")
    dec2 = pd.qcut(df2["coverage_max"], 10, labels=False, duplicates="drop")
    ll_dec = df2.groupby(dec2).apply(
        lambda g: logloss(g["is_correct"], g["oof_group"]), include_groups=False)
    R.append(md_table(["coverage decile", "median cov", "rows", "target rate", "TF-IDF baseline log loss"],
                      [(int(i), f"{df.loc[dec == i, 'coverage_max'].median():.3f}",
                        f"{r['size']:,.0f}", f"{r['mean']:.3f}", f"{ll_dec[i]:.4f}")
                       for i, r in dd.iterrows()]) + "\n")

    # lowest-coverage examples with best-matching window
    R.append("**Lowest-coverage rows** (objective vs best-matching window)\n")
    ex_rows = []
    worst = df.nsmallest(6, "coverage_max")
    words_cache = {}
    with Pool(2, initializer=_init_worker, initargs=(str(tdir),)) as pool:
        for sid, w in pool.map(read_session_words, worst["session_id"].tolist()):
            words_cache[sid] = w
    for _, rw in worst.iterrows():
        a, b = bounds[rw["session_id"]]
        segs = emb_seg[a:b]
        k = int(np.argmax(segs @ emb_query[lo_ix[rw["learning_objective_id"]]]))
        w = words_cache[rw["session_id"]]
        snippet = " ".join(w[k * SEG_STRIDE: k * SEG_STRIDE + 40])
        ex_rows.append((rw["session_id"], f"{rw['coverage_max']:.3f}", int(rw["is_correct"]),
                        rw["learning_objective"][:55], snippet[:110].replace("|", "/")))
    R.append(md_table(["session", "cov", "label", "objective", "best window (first 40 words)"], ex_rows) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(c, bins=60, color=P["blue"], edgecolor=P["surface"], lw=.4)
    axes[0].set_xlabel("coverage_max (objective ↔ best window cosine)")
    axes[0].set_ylabel("rows"); axes[0].set_title("Coverage distribution")
    med = [df.loc[dec == i, "coverage_max"].median() for i in dd.index]
    err = 1.96 * np.sqrt(dd["mean"] * (1 - dd["mean"]) / dd["size"])
    axes[1].axhline(prior, color=P["baseline"], ls="--", lw=1)
    axes[1].errorbar(med, dd["mean"], yerr=err, color=P["blue"], lw=2, marker="o",
                     ms=4, capsize=2, ecolor=P["baseline"])
    axes[1].set_xlabel("coverage decile (median)"); axes[1].set_ylabel("target rate")
    axes[1].set_title("Objective coverage → correctness")
    for ax in axes:
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(figdir / "e02_coverage_vs_target.png"); plt.close(fig)

    # ================================================== 3. datamaps (UMAP)
    R.append("## 3. Datamaps\n")
    import umap
    print("[umap] objectives ...")
    u_obj = umap.UMAP(n_neighbors=12, min_dist=0.25, metric="cosine",
                      random_state=RNG_SEED).fit_transform(emb_obj)
    print("[umap] sessions ...")
    sess_emb = np.vstack([emb_seg[a:b].mean(0) if b > a else np.zeros(emb_seg.shape[1])
                          for a, b in (bounds[s] for s in sids)])
    sess_emb /= np.maximum(np.linalg.norm(sess_emb, axis=1, keepdims=True), 1e-9)
    u_sess = umap.UMAP(n_neighbors=25, min_dist=0.15, metric="cosine",
                       random_state=RNG_SEED).fit_transform(sess_emb)

    from matplotlib.colors import LinearSegmentedColormap
    cmap_blue = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    top7 = tt.sort_values("size", ascending=False).head(7).index.tolist()
    col_of = {t: SERIES[i] for i, t in enumerate(top7)}
    cols = [col_of.get(t, P["baseline"]) for t in lo["topic_emb_label"]]
    axes[0].scatter(u_obj[:, 0], u_obj[:, 1], s=22, c=cols, linewidths=0, alpha=.9)
    for t in top7:
        m = (lo["topic_emb_label"] == t).to_numpy()
        cx, cy = u_obj[m, 0].mean(), u_obj[m, 1].mean()
        axes[0].text(cx, cy, t.split(",")[0], fontsize=8, fontweight="bold",
                     color=P["ink"], ha="center",
                     bbox=dict(fc=P["surface"], ec="none", alpha=.75, pad=1))
    axes[0].set_title("398 objectives — embedding topics (top-7 colored)")
    sc = axes[1].scatter(u_obj[:, 0], u_obj[:, 1], s=22,
                         c=(lo["size"] * lo["mean"] + 30 * prior) / (lo["size"] + 30),
                         cmap=cmap_blue, linewidths=0, alpha=.9)
    fig.colorbar(sc, ax=axes[1], shrink=.8, label="smoothed target rate")
    axes[1].set_title("Same map — objective difficulty")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(figdir / "e01_objective_datamap.png"); plt.close(fig)

    sess_rate = df.groupby("session_id")["is_correct"].mean().reindex(sids).to_numpy()
    sess_topic = (d2.groupby("session_id")["topic_emb_label"].agg(lambda s: s.iloc[0])
                    .reindex(sids))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    cols = [col_of.get(t, P["baseline"]) for t in sess_topic]
    axes[0].scatter(u_sess[:, 0], u_sess[:, 1], s=3, c=cols, linewidths=0, alpha=.5)
    axes[0].set_title("22,821 sessions — colored by objective topic")
    sc = axes[1].scatter(u_sess[:, 0], u_sess[:, 1], s=3, c=sess_rate, cmap=cmap_blue,
                         linewidths=0, alpha=.6)
    fig.colorbar(sc, ax=axes[1], shrink=.8, label="session mean label")
    axes[1].set_title("Same map — session outcome")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.tight_layout(); fig.savefig(figdir / "e03_session_datamap.png"); plt.close(fig)

    r_topic = float(pd.Series(cols).ne(P["baseline"]).mean())
    R.append(f"- `e01_objective_datamap.png` — objective map: topic islands + difficulty gradient.")
    R.append(f"- `e03_session_datamap.png` — session map ({r_topic:.0%} of sessions in a top-7 "
             f"topic color). If sessions cluster by *content*, the topic colors form islands; "
             f"outcome color shows whether any content region is systematically hard.\n")

    R.append("## Figures\n")
    for f in sorted(figdir.glob("*.png")):
        R.append(f"![{f.stem}](figures/{f.name})\n")
    R.append(f"\n_Runtime: {time.time()-t_start:.0f}s_\n")
    (out / "REPORT.md").write_text("\n".join(R), encoding="utf-8")
    print(f"Done. {out/'REPORT.md'} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()