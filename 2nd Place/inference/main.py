#!/usr/bin/env python
"""Trace the Ace — inference entry point (runs at the submission root).

Tabular blend of TF-IDF+LogisticRegression (fold-bagged) and LightGBM
(fold-bagged); blend weight, clipping and feature order from assets/config.json.
If assets/llm_stack.json + assets/llm/ exist (SUB-6+), a Qwen3 LoRA transcript
reader is scored via vLLM and stacked on the tabular logit (fit_sub6.py); any
failure in that stage falls back to the tabular probabilities — unless
ACE_REQUIRE_LLM=1 (strict mode for container smoke tests: re-raise instead of
masking, so a broken LLM path cannot pass silently). ACE_SKIP_LLM=1 skips the
stage entirely.

Expects (provided by the runtime):
    data/test_features.csv, data/test_transcripts/<session_id>.csv, data/submission_format.csv
    assets/  (see fit_final_v2.py / stage_llm.sh for contents)
Writes:
    submission.csv  (response_id, probability)

NOTE (competition logging rule): this script prints stage markers only — never
counts, statistics, or content derived from the test data.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _configure_vllm_runtime_env() -> None:
    """Must run before any vLLM import (module top covers spawned workers too).

    The official runtime image has the CUDA runtime but no nvcc; vLLM's
    FlashInfer sampler ("Using FlashInfer for top-p & top-k sampling") may JIT
    kernels with nvcc on some GPUs -> "/usr/local/cuda/bin/nvcc: not found",
    which killed the LLM stage in the container smoke test. Force the native
    sampler path (VLLM_USE_FLASHINFER_SAMPLER parses via bool(int(...)) — same
    contract in the runtime's vLLM 0.21 and current versions).
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")     # no-network safety
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_configure_vllm_runtime_env()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "assets"))

import numpy as np
import pandas as pd
from scipy.sparse import hstack, diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from ace.features import derive_transcript_columns
from ace.transcripts import RAW_STAT_KEYS, lexical_overlap, parse_transcript, segment_text

ASSETS = ROOT / "assets"
DATA = ROOT / "data"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tfidf_transform(texts, npz_path: Path, ngram_range):
    """Version-proof TF-IDF transform from raw vocab+idf arrays (sublinear tf, l2 norm)."""
    z = np.load(npz_path, allow_pickle=True)
    terms, idf = z["terms"], z["idf"]
    vocab = {t: i for i, t in enumerate(terms)}
    cv = CountVectorizer(vocabulary=vocab, ngram_range=ngram_range)
    X = cv.transform(texts).astype(np.float64)
    X.data = 1.0 + np.log(X.data)          # sublinear tf
    X = X @ diags(idf)
    return normalize(X, norm="l2", copy=False)


def tabular_v2_predict(df, tabv2_dir, bge_model):
    """SUB-9 structured transcript branch: test-time twin of ace.models.train_tabular_v2.

    Rebuilds the same feature frame from the test transcripts and averages the
    fold x seed boosters. Two things MUST match the offline fit or the logit scale
    silently drifts and the stack coefficient mis-weights this branch:
      * the feature LIST and ORDER come from the saved config (`features`), not from
        recomputing feature_columns() here -- a future ACE feature change would
        otherwise reorder the LightGBM input;
      * `te_lo` is ABSENT by construction (the shipped fit used --no-te-lo, because
        the objective target encoding is anti-predictive on unseen skills, E21/E14).
        If a future config reintroduces it, it must also be recomputed here.
    """
    import lightgbm as lgb

    from ace.tabular_v2_features import (
        TABULAR_V2_BGE_FEATURES,
        TABULAR_V2_RESPONSE_LEXICAL_FEATURES,
        TABULAR_V2_SESSION_FEATURES,
        add_derived_session_features,
        build_response_lexical_features,
        build_session_feature_table,
        build_student_bge_coverage,
    )

    cfg = json.loads((tabv2_dir / "config.json").read_text())
    cols = list(cfg["features"])
    assert "te_lo" not in cols, "shipped tabular_v2 must be the te_lo-free fit (E21)"

    tdir = DATA / "test_transcripts"
    workers = min(16, os.cpu_count() or 4)
    sids = sorted(df["session_id"].astype(str).unique())

    session = add_derived_session_features(
        build_session_feature_table(tdir, session_ids=sids, workers=workers))
    feats = df[["response_id", "session_id", "learning_objective"]].copy()
    lexical = build_response_lexical_features(feats, tdir, workers=workers)

    # Select ONLY the named feature columns before merging, exactly as
    # train_tabular_v2._load_training_frame does. session_features.csv shares names
    # with ACE's base transcript columns (n_utt, w_total, dur_s, ...); merging the whole
    # table suffixes both sides to _x/_y and the base features silently disappear.
    sc = [c for c in TABULAR_V2_SESSION_FEATURES if c in session.columns]
    frame = df.merge(session[["session_id"] + sc], on="session_id",
                     how="left", validate="m:1")
    lc = [c for c in TABULAR_V2_RESPONSE_LEXICAL_FEATURES if c in lexical.columns]
    frame = frame.merge(lexical[["response_id"] + lc], on="response_id",
                        how="left", validate="1:1")
    if any(c in cols for c in TABULAR_V2_BGE_FEATURES):
        # Pass the LIVE encoder, never a path: constructing a second SentenceTransformer
        # here would double VRAM immediately before the vLLM engine loads.
        bge = build_student_bge_coverage(feats, tdir, bge_model, show_progress=False)
        bc = [c for c in TABULAR_V2_BGE_FEATURES if c in bge.columns]
        frame = frame.merge(bge[["response_id"] + bc], on="response_id",
                            how="left", validate="1:1")

    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"tabular_v2 features absent at inference: {missing}")
    dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if dupes:
        # a suffixed/duplicated merge column would silently feed the wrong values
        raise RuntimeError(f"tabular_v2 duplicate columns after merge: {dupes}")
    n_expected = int(cfg["n_features"])
    if not (len(cols) == n_expected == 172):
        raise RuntimeError(f"tabular_v2 feature count {len(cols)} != {n_expected} (expected 172)")
    X = frame[cols].replace([np.inf, -np.inf], np.nan)

    models = sorted(tabv2_dir.glob("models/*.txt"))
    n_models = int(cfg["n_models"])
    if len(models) != n_models:
        raise RuntimeError(f"tabular_v2 expected {n_models} boosters, found {len(models)}")
    preds = []
    for f in models:
        bst = lgb.Booster(model_file=str(f))
        # LightGBM consumes columns POSITIONALLY: an ordering drift would feed each
        # value into the wrong split thresholds and still return plausible numbers.
        if bst.feature_name() != cols:
            raise RuntimeError(f"tabular_v2 feature ordering mismatch in {f.name}")
        preds.append(bst.predict(X))  # binary objective -> probabilities, not raw scores
    p = np.mean(preds, axis=0)  # average PROBABILITIES across the fold x seed boosters
    if not np.isfinite(p).all():
        raise RuntimeError("tabular_v2 produced non-finite predictions")
    # logit_scale 1.0 == uncorrected. The ensemble-vs-single-booster logit scale is not
    # cleanly measurable (no holdout exists that all 15 boosters were held out from), so
    # nothing is applied rather than a correction estimated from contaminated predictions.
    s = float(cfg.get("logit_scale", 1.0))
    if s != 1.0:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        p = 1.0 / (1.0 + np.exp(-(s * np.log(pc / (1 - pc)))))
    return np.clip(p, 1e-6, 1 - 1e-6)  # same epsilon convention as the tabular pipeline


def build_context_index(df, bundle, recipe, out_path):
    """Build the label-free BGE objective-context index this reader's prompt needs.

    Only invoked for a bundle whose recipe declares prompt_mode=objective_chunks
    (SUB-10 / OLMo). Runs as its own subprocess so the sentence-transformers CUDA
    context is fully released before any vLLM engine loads — the same isolation
    rationale as the per-bundle score workers. Uses only each row's own objective
    and transcript; it never sees a label, and BGE is already staged for tabular_v2.
    """
    import subprocess
    import tempfile

    bge = ASSETS / "bge-base-en-v1.5"
    if not bge.exists():
        raise RuntimeError(f"prompt_mode=objective_chunks needs BGE at {bge}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ASSETS)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    with tempfile.TemporaryDirectory() as tmp:
        feats_csv = Path(tmp) / "ctx_features.csv"
        df[["response_id", "session_id", "learning_objective"]].to_csv(feats_csv, index=False)
        subprocess.run(
            [sys.executable, "-m", "ace.llm.prepare_objective_contexts",
             "--features", str(feats_csv),
             "--transcripts-dir", str(DATA / "test_transcripts"),
             "--out", str(out_path),
             "--bge-model", str(bge),
             "--serialize", str(int(recipe["serialize"])),
             "--window-words", str(int(recipe.get("context_window_words", 240))),
             "--stride-words", str(int(recipe.get("context_stride_words", 120))),
             "--top-k", str(int(recipe.get("context_top_k", 8)))],
            check=True, env=env)


def bundle_recipe(stack, bundle):
    """Per-bundle prompt recipe, falling back to the stack-level defaults.

    Readers in one stack need not share a prompt: SUB-10 pairs two 8K
    full-transcript Qwen readers with a 4K compressed-key-moment OLMo reader.
    Anything absent from a bundle's "recipe" inherits the stack value, so the
    4B/8B entries stay byte-identical to SUB-9.
    """
    r = dict(stack.get("recipe", {}))
    r.update(bundle.get("recipe", {}))
    return {"serialize": int(r.get("serialize", stack.get("serialize", 1))),
            "max_input_tokens": int(r.get("max_input_tokens",
                                          stack.get("max_input_tokens", 8192))),
            "prompt_mode": str(r.get("prompt_mode", "standard")),
            "context_full_threshold": int(r.get("context_full_threshold", 0)),
            "context_chunks": int(r.get("context_chunks", 2)),
            "context_head_frac": float(r.get("context_head_frac", 0.16)),
            "context_tail_frac": float(r.get("context_tail_frac", 0.34)),
            "context_window_words": int(r.get("context_window_words", 240)),
            "context_stride_words": int(r.get("context_stride_words", 120)),
            "context_top_k": int(r.get("context_top_k", 8))}


def reader_base_ready(bundle_dir, runtime_root=None) -> bool:
    """Is this reader's base model reachable? MODE-AWARE, not a path guess.

    `(<bundle>/base).exists()` is right for an embedded bundle and WRONG for an
    external one: an external bundle deliberately ships no local `base/` and names a
    model the runtime preloads under `huggingface_models/{org}/{model}/`, so the old
    test classified a perfectly good reader as "not staged" and dropped it.

    A malformed `bundle.json` RAISES rather than returning False — a broken bundle must
    fail the job, never quietly disable a reader the stack declares. That is the same
    required-reader policy the rest of this stage enforces.
    """
    from ace.llm.bundle import base_is_available
    return base_is_available(bundle_dir, runtime_root=runtime_root)


def manual_blends(stack):
    """Optional explicitly-configured logit blends between two reader features.

    Absent from every stack shipped so far, and a no-op when absent.

    WHY THIS EXISTS. A reader whose cross-validation is still running cannot have a
    fitted stack coefficient — fitting one on the folds that happen to have finished
    and calling it validated is the exact error the SUB-11 post-mortem records. But it
    can still be probed at a small, deliberately chosen weight. This mechanism makes
    that weight an explicit number in the stack json:

        "manual_blend": [{"target": "z_ft_8b", "source": "z_ft_q35", "w": 0.25}]

    means, before the stack's linear combination is applied:

        z_target := (1 - w) * z_target + w * z_source

    so the fitted coefficient on `target` is reused unchanged and the probe cannot
    change the stack's shape. `source` is scored like any other bundle but need not
    appear in `features`.

    Deliberately generic: it names features, not architectures. `w` must be written
    down by a human in the config — it is never inferred from test predictions and
    never calibrated from leaderboard feedback.
    """
    spec = stack.get("manual_blend") if stack else None
    if not spec:
        return []
    blends = spec if isinstance(spec, list) else [spec]
    feats = set(stack.get("features", []))
    out = []
    for b in blends:
        target, source, w = b.get("target"), b.get("source"), b.get("w")
        if target not in feats:
            raise RuntimeError(f"manual_blend target {target!r} is not a stack feature")
        if not isinstance(w, (int, float)) or not 0.0 <= float(w) <= 1.0:
            raise RuntimeError(f"manual_blend w must be a number in [0,1], got {w!r}")
        if not source:
            raise RuntimeError("manual_blend requires a source feature")
        out.append({"target": target, "source": source, "w": float(w)})
    return out


def apply_manual_blends(vals, blends):
    """In-place `z_target := (1-w)*z_target + w*z_source` for each configured blend."""
    for b in blends:
        for key in (b["target"], b["source"]):
            if key not in vals:
                raise RuntimeError(f"manual_blend needs {key!r} but it was not scored")
        w = b["w"]
        vals[b["target"]] = (1.0 - w) * vals[b["target"]] + w * vals[b["source"]]
        print(f"[6/6] manual blend applied: {b['target']} <- "
              f"{1 - w:.3f}*{b['target']} + {w:.3f}*{b['source']}")
    return vals


def llm_z_ft(df, stack, bundle_dir, recipe=None, context_index=None):
    """Transcript-reader logit per row for ONE LLM bundle, via an isolated subprocess.

    Each bundle (4B, 8B, ...) is scored in a fresh `ace.llm.score_worker` process
    so its vLLM engine gets a clean CUDA/process lifecycle and exits fully before
    the next bundle loads — a two-engine in-process teardown is unreliable. The
    worker writes (response_id, z_ft); we read it back aligned to df's row order.

    z_ft = mean logit over k adapters; k adapts to test size via the row-pass
    budget (k=1 for a single full-fit adapter, matching the single-adapter OOF
    regime the stacker was calibrated on).
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rows_csv = Path(tmp) / "rows.csv"
        out_csv = Path(tmp) / "z_ft.csv"
        df[["response_id", "session_id", "learning_objective"]].to_csv(rows_csv, index=False)
        # the worker is a fresh interpreter: it must find the bundled `ace` package.
        # prepend assets/ so an existing PYTHONPATH (runtime venv) is preserved.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ASSETS)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        r = recipe if recipe is not None else bundle_recipe(stack, {})
        cmd = [sys.executable, "-m", "ace.llm.score_worker",
               "--bundle", str(bundle_dir),
               "--rows", str(rows_csv),
               "--transcripts", str(DATA / "test_transcripts"),
               "--out", str(out_csv),
               "--serialize", str(r["serialize"]),
               "--max-input-tokens", str(r["max_input_tokens"]),
               "--rowpass-budget", str(int(stack.get("rowpass_budget", 50_000)))]
        if r["prompt_mode"] != "standard":
            cmd += ["--prompt-mode", r["prompt_mode"],
                    "--context-full-threshold", str(r["context_full_threshold"]),
                    "--context-chunks", str(r["context_chunks"]),
                    "--context-head-frac", str(r["context_head_frac"]),
                    "--context-tail-frac", str(r["context_tail_frac"])]
            if context_index is not None:
                cmd += ["--context-index", str(context_index)]
        subprocess.run(cmd, check=True, env=env)
        z = pd.read_csv(out_csv)
    # align to df order (worker preserves response_id, not necessarily row order)
    return df[["response_id"]].merge(z, on="response_id", how="left")["z_ft"].to_numpy()


def main():
    print("[1/6] load inputs")
    feats = pd.read_csv(DATA / "test_features.csv")
    feats["learning_objective"] = feats["learning_objective"].str.strip()
    sub = pd.read_csv(DATA / "submission_format.csv")
    cfg = json.loads((ASSETS / "config.json").read_text())

    print("[2/6] parse transcripts")
    sids = sorted(feats["session_id"].unique())
    workers = min(16, os.cpu_count() or 4)
    with Pool(workers) as pool:
        parsed = pool.map(parse_transcript,
                          [str(DATA / "test_transcripts" / f"{s}.csv") for s in sids],
                          chunksize=32)
    stats_by_sid = {sid: st for sid, st, _, _ in parsed}
    text_by_sid = {sid: tx for sid, _, tx, _ in parsed}
    toks_by_sid = {sid: tk for sid, _, _, tk in parsed}

    df = feats.copy()
    ts = pd.DataFrame([{"session_id": s, **stats_by_sid[s]} for s in sids])
    df = df.merge(derive_transcript_columns(ts), on="session_id", how="left")
    df["lo_overlap"] = [lexical_overlap(t, toks_by_sid.get(s, set()))
                        for t, s in zip(df["learning_objective"], df["session_id"])]
    df["lo_words"] = df["learning_objective"].str.split().str.len()

    print("[3/6] text model")
    Xt_sess = tfidf_transform([text_by_sid[s] for s in sids],
                              ASSETS / "tfidf_transcript.npz", (1, 2))
    sid_ix = {s: i for i, s in enumerate(sids)}
    Xt = Xt_sess[[sid_ix[s] for s in df["session_id"]]]
    Xo = tfidf_transform(df["learning_objective"].tolist(),
                         ASSETS / "tfidf_objective.npz", (1, 2))
    X_text = hstack([Xt, Xo * cfg["objective_weight"]]).tocsr()
    lr = np.load(ASSETS / "logreg.npz")
    logits = X_text @ lr["coefs"].T.astype(np.float64) + lr["intercepts"]
    p_text = (1.0 / (1.0 + np.exp(-logits))).mean(axis=1)

    print("[4/6] coverage + objective features")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(ASSETS / "bge-base-en-v1.5"), device=device)
    lo_unique = df["learning_objective"].drop_duplicates().tolist()
    emb_q = model.encode([BGE_QUERY_PREFIX + t for t in lo_unique], batch_size=64,
                         normalize_embeddings=True, convert_to_numpy=True)
    q_ix = {t: i for i, t in enumerate(lo_unique)}
    segments, bounds = [], {}
    for s in sids:
        segs = segment_text(text_by_sid[s])
        bounds[s] = (len(segments), len(segments) + len(segs))
        segments.extend(segs)
    emb_seg = model.encode(segments, batch_size=256, normalize_embeddings=True,
                           convert_to_numpy=True)
    cov_max = np.full(len(df), np.nan)
    cov_top3 = np.full(len(df), np.nan)
    for i, (s, t) in enumerate(zip(df["session_id"], df["learning_objective"])):
        a, b = bounds[s]
        if b <= a:
            continue
        sim = emb_seg[a:b] @ emb_q[q_ix[t]]
        cov_max[i] = sim.max()
        cov_top3[i] = np.sort(sim)[-3:].mean()
    df["coverage_max"], df["coverage_top3"] = cov_max, cov_top3
    df["coverage_gap"] = df["coverage_max"] - df["coverage_top3"]

    z = np.load(ASSETS / "te_objective.npz", allow_pickle=True)
    te_map = dict(zip(z["lo_ids"], z["te"]))
    cnt_map = dict(zip(z["lo_ids"], z["cnt"]))
    prior = float(z["prior"][0])
    df["te_lo"] = df["learning_objective_id"].map(te_map)
    df["lo_count"] = df["learning_objective_id"].map(cnt_map).fillna(0.0)
    # unknown objective ids: difficulty via embedding-kNN over the train objectives
    # (config v2; the test set behaves as ~100% unseen objectives — see EXPERIMENTS.md E11)
    # SUB-11 sim gate input: how close is this row's objective to ANYTHING the models
    # trained on? An objective present in the training table is its own nearest
    # neighbour, so 1.0; an unseen one takes its best cosine over the train objectives —
    # the same `sims` matrix the kNN backoff below already builds, so the gate costs
    # nothing extra. Defaulting to 1.0 means "not novel": if embeddings are unavailable
    # the gate simply never fires and the pipeline behaves exactly as SUB-9.
    df["max_sim"] = 1.0
    unk = df["te_lo"].isna()
    if unk.any() and "emb" in z:
        knn = cfg.get("knn", {"k": 10, "a_knn": 10.0, "a_prior": 10.0})
        emb_train = z["emb"].astype(np.float32)
        rate, cnt = z["rate"], z["cnt"].astype(float)
        texts = df.loc[unk, "learning_objective"].drop_duplicates().tolist()
        emb_u = model.encode(texts, batch_size=64, normalize_embeddings=True,
                             convert_to_numpy=True)  # plain, same space as emb_objectives
        sims = emb_u @ emb_train.T
        idx = np.argsort(-sims, axis=1)[:, :knn["k"]]
        s = np.take_along_axis(sims, idx, axis=1).clip(min=0)
        w = s * cnt[idx]
        r_knn = (w * rate[idx]).sum(1) / np.maximum(w.sum(1), 1e-9)
        te_u = (knn["a_knn"] * r_knn + knn["a_prior"] * prior) / (knn["a_knn"] + knn["a_prior"])
        df.loc[unk, "te_lo"] = df.loc[unk, "learning_objective"].map(dict(zip(texts, te_u)))
        df.loc[unk, "max_sim"] = df.loc[unk, "learning_objective"].map(
            dict(zip(texts, sims.max(axis=1).astype(float))))
    df["max_sim"] = df["max_sim"].astype(float).fillna(1.0)
    df["te_lo"] = df["te_lo"].fillna(prior)
    topic_map = json.loads((ASSETS / "topic_map.json").read_text())
    df["topic_emb"] = df["learning_objective_id"].map(topic_map).fillna(-1).astype(int)

    # SUB-9: structured transcript branch (tabular_v2). Computed HERE, while the bge
    # encoder is still resident, because the student-coverage features need it and
    # reloading it after the vLLM engine starts would contend for VRAM. Produces a
    # probability consumed as `z_tabv2` by the stack below; None disables the feature
    # (any failure degrades to the previously shipped stack rather than losing the run).
    # The stack json is the single source of truth for which readers are REQUIRED.
    # A reader the stack declares but cannot run must fail the job: a run that quietly
    # drops one scores a different model than the one that was validated offline, and
    # exits 0 while doing it (how SUB-6's FlashInfer bug presented).
    p_tabv2 = None
    tabv2_dir = ASSETS / "tabular_v2"
    _sp = ASSETS / "llm_stack.json"
    _stack_feats = json.loads(_sp.read_text()).get("features", []) if _sp.exists() else []
    if "z_tabv2" in _stack_feats:
        if os.environ.get("ACE_SKIP_TABV2"):
            raise RuntimeError("ACE_SKIP_TABV2 set but llm_stack.json requires z_tabv2")
        if not tabv2_dir.exists():
            raise RuntimeError(f"llm_stack.json requires z_tabv2 but {tabv2_dir} is absent")
        print("[4/6] running tabular_v2")
        p_tabv2 = tabular_v2_predict(df, tabv2_dir, model)  # no swallow: must raise
        print("[4/6] tabular_v2 complete")

    del model  # release the bge encoder before the (optional) vLLM engine
    torch.cuda.empty_cache()

    print("[5/6] GBM + tabular blend")
    import lightgbm as lgb
    X_gbm = df[cfg["gbm_features"]]
    model_files = sorted((ASSETS / "gbm").glob("*.txt"))  # 5 (v1) or seeds x 5 (v2)
    preds = [lgb.Booster(model_file=str(f)).predict(X_gbm) for f in model_files]
    p_gbm = np.mean(preds, axis=0)

    w = cfg["blend_w_gbm"]
    lo_clip, hi_clip = cfg["clip"]

    def calibrate(p_raw, platt):
        """Shrink -> Platt -> clip. Both blend arms get identical treatment."""
        q = np.clip(p_raw, 1e-6, 1 - 1e-6)
        # logit shrinkage toward the train prior: LB calibration for the weaker test
        # signal (SUB-2: local AUC 0.737 realized as 0.6135 -> logits ~2.2x too spread)
        a = cfg.get("shrink", 1.0)
        if a != 1.0:
            zs = a * np.log(q / (1 - q)) + (1 - a) * np.log(prior / (1 - prior))
            q = 1.0 / (1.0 + np.exp(-zs))
        # Platt recalibration fit on the unseen-objective-CV OOF (config v2)
        pa, pb = platt
        if (pa, pb) != (1.0, 0.0):
            q = 1.0 / (1.0 + np.exp(-(pa * np.log(q / (1 - q)) + pb)))
        return np.clip(q, lo_clip, hi_clip)

    p = calibrate(w * p_gbm + (1 - w) * p_text, cfg.get("platt", [1.0, 0.0]))

    # SUB-11 sim-gated blend (E12/E14). te_lo's kNN backoff borrows difficulty from
    # unrelated skills, so on NOVEL objectives the history-bearing GBM goes
    # anti-predictive (E14: novel-band AUC 0.5875 vs 0.6083 without it). On those rows
    # only, swap in a history-free blend with its own weight and Platt. Confirmed on all
    # three protocol-v3 fold seeds and in the frozen SUB-9 stack (dAUC +0.0017 at 100%).
    # Self-limiting by construction: no novel rows -> byte-identical to SUB-9.
    gate = cfg.get("sim_gate")
    if gate:
        nh_dir = ASSETS / "gbm_nh"
        nh_files = sorted(nh_dir.glob("*.txt")) if nh_dir.is_dir() else []
        if not nh_files:
            # Same required-asset policy as the LLM readers: a silent fallback would
            # emit a valid-looking submission from a different model than was validated.
            raise RuntimeError(f"config declares sim_gate but no boosters at {nh_dir}")
        p_gbm_nh = np.mean([lgb.Booster(model_file=str(f)).predict(df[gate["gbm_nh_features"]])
                            for f in nh_files], axis=0)
        w_nh = gate["blend_w_gbm_nh"]
        p_nh = calibrate(w_nh * p_gbm_nh + (1 - w_nh) * p_text, gate["platt_nh"])
        novel = df["max_sim"].to_numpy(float) < gate["sim_threshold"]
        p = np.where(novel, p_nh, p)
        print(f"[5/6] sim-gated blend active (threshold {gate['sim_threshold']}, "
              f"{len(nh_files)} history-free boosters)")

    # SUB-6+: stack one or more LLM transcript readers on the tabular logit
    # (fit_sub6.py / the 2-LLM extension). The stack json declares its inputs:
    #   single-LLM (SUB-6): features ["z_blend","z_ft"], bundle at assets/llm/
    #   multi-LLM (SUB-7+): a "bundles" list mapping each z_ft* feature to an
    #     assets/<dir>; each is scored in its own subprocess (main.py llm_z_ft).
    # REQUIRED-READER POLICY: every reader the stack declares is mandatory. A silent
    # fallback would emit a valid-looking submission scored by a DIFFERENT model than the
    # one validated offline, at exit 0 — the SUB-6 FlashInfer failure mode. A failed job
    # is preferable. Legacy stacks that declare no reader features still fall back.
    stack_path = ASSETS / "llm_stack.json"
    stack = json.loads(stack_path.read_text()) if stack_path.exists() else None
    bundles = (stack.get("bundles", [{"feature": "z_ft", "dir": "llm"}])
               if stack else [])
    req_feats = list(stack["features"]) if stack else []
    # A manual-blend SOURCE is a reader the run genuinely depends on even though it is
    # not itself a stack feature, so it inherits the same required-reader policy: a
    # missing one must fail the job, not quietly leave the target feature unblended.
    blends = manual_blends(stack)
    req_feats += [b["source"] for b in blends if b["source"] not in req_feats]
    llm_required = [f for f in req_feats if f.startswith("z_ft")]

    def _reader_ready(b) -> bool:
        return reader_base_ready(ASSETS / b["dir"], runtime_root=ROOT)

    if llm_required:
        if os.environ.get("ACE_SKIP_LLM"):
            raise RuntimeError(f"ACE_SKIP_LLM set but the stack requires {llm_required}")
        absent = [b["dir"] for b in bundles
                  if b["feature"] in req_feats and not _reader_ready(b)]
        if absent:
            raise RuntimeError(
                f"stack requires readers whose base model is unavailable: {absent} "
                f"(embedded bundles need assets/<dir>/base/; external bundles need the "
                f"runtime to provide their declared external_base)")
        unbundled = [f for f in llm_required
                     if f not in {b["feature"] for b in bundles}]
        if unbundled:
            raise RuntimeError(f"stack features with no bundle mapping: {unbundled}")

    if stack and (llm_required or not os.environ.get("ACE_SKIP_LLM")) \
            and all(_reader_ready(b) for b in bundles):
        print("[6/6] transcript reader(s) (LLM) + stack")
        try:
            vals = {"z_blend": np.log(p / (1 - p))}  # p is clipped, logit is finite
            if p_tabv2 is not None:
                pt = np.clip(p_tabv2, 1e-6, 1 - 1e-6)
                vals["z_tabv2"] = np.log(pt / (1 - pt))
            import tempfile
            with tempfile.TemporaryDirectory() as ctx_tmp:
                for b in bundles:
                    r = bundle_recipe(stack, b)
                    idx = None
                    if r["prompt_mode"] == "objective_chunks":
                        # Built here, inside the reader stage, so a failure lands in
                        # the required-reader path below rather than killing the run
                        # before the tabular result exists.
                        print(f"[6/6] building objective context index for {b['dir']}")
                        idx = Path(ctx_tmp) / f"{b['dir']}_objctx.jsonl"
                        build_context_index(df, b, r, idx)
                    vals[b["feature"]] = llm_z_ft(df, stack, ASSETS / b["dir"],
                                                  recipe=r, context_index=idx)
            # Applied BEFORE the linear combination, so the fitted coefficients are
            # reused exactly and a probe reader cannot alter the stack's shape.
            apply_manual_blends(vals, blends)
            missing = [f for f in stack["features"] if f not in vals]
            assert not missing, f"stack features with no input: {missing}"
            z = float(stack["intercept"]) + sum(
                c * vals[f] for c, f in zip(stack["coef"], stack["features"]))
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, *stack.get("clip", [lo_clip, hi_clip]))
            print(f"[6/6] {len(stack['features'])}-feature stack applied")
        except Exception as e:  # noqa: BLE001
            if llm_required or os.environ.get("ACE_REQUIRE_LLM", "0") == "1":
                raise  # a declared reader failed: fail the job, never substitute a model
            # legacy stacks that declare no reader features keep the old fallback
            print(f"[6/6] LLM stage FAILED ({type(e).__name__}) — tabular fallback",
                  flush=True)

    out = sub[["response_id"]].merge(
        pd.DataFrame({"response_id": df["response_id"], "probability": p}),
        on="response_id", how="left")
    out["probability"] = out["probability"].fillna(prior)
    assert list(out.columns) == ["response_id", "probability"]
    assert len(out) == len(sub)
    out.to_csv(ROOT / "submission.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()