"""Submission entrypoint for the Trace the Ace code-execution runtime.

The runtime unzips this submission into ``/code_execution/``, runs
``python main.py`` with the working directory there, mounts the test data
read-only at ``./data``, and copies the resulting ``./submission.csv`` back out.
No internet access at inference; everything needed must ship in ``assets/``, except the
base Qwen3-8B weights, which the runtime mounts.

Zip layout of the submission (``assets/model_kind.txt`` = ``ensemble``):

    main.py
    main_llmkt_lora.py              # LLM-only entry point; not run (see its docstring)
    assets/
        manifest_sha256.txt         # sha256 of every other file in assets/, checked on startup
        model_kind.txt              # "ensemble": the fine-tuned LLM blended with the XGBoost model
        blend_weight.txt            # XGBoost weight w of the blend (0.4300)
        lora_llm/                   # LoRA adapter of the LLM half, merged onto the base
        lora_sha256.txt             # sha256 of the two adapter files
        tokenizer/                  # Qwen3 tokenizer files
        llmkt_calibrator.joblib     # frozen Platt map of the LLM half
        baseline.joblib             # the XGBoost half (seed ensemble + frozen Platt map)
        *_clf.joblib                # six distilled per-turn classifiers (feature blocks)
        v2_spec.txt                 # which taxonomy-v2 block the XGBoost model was trained on
        nc_curriculum.csv           # National Curriculum statements (objective features)
        magnificat/                 # vendored feature/data code (single source of truth)

Other values of ``model_kind.txt`` select other feature and model paths in ``main()``.

Test data layout the runtime provides:

    data/
        test_features.csv          # response_id, session_id, learning_objective_id, learning_objective
        submission_format.csv      # response_id, probability  (rows we must output)
        test_transcripts/<sid>.csv # one transcript per session (a DIRECTORY, not a zip)
"""

import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the runtime log clean (500-line cap)

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
sys.path.insert(0, str(ASSETS))  # make the vendored `magnificat` importable

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

DATA_DIR = Path("data")
SUBMISSION_PATH = Path("submission.csv")
MODEL_PATH = ASSETS / "baseline.joblib"
KIND_PATH = ASSETS / "model_kind.txt"

#: kinds that blend the XGBoost model with the fine-tuned LLM. The submission uses `ensemble`.
ENSEMBLE_KINDS = ("ensemble", "ensemble_lk")
#: `model_e3l`: the XGBoost half of `ensemble_lk` alone.
GBM_LK_KINDS = ("model_e3l",)
#: every kind that builds the full E3L feature matrix (features only -- says nothing about
#: whether an LLM also runs)
E3L_FEATURE_KINDS = (*ENSEMBLE_KINDS, *GBM_LK_KINDS)
MANIFEST_PATH = ASSETS / "manifest_sha256.txt"
EPS = 1e-6  # clip probabilities so log loss never sees exactly 0 or 1


def _booster_selects(prefix):
    """True if the staged booster actually selects any column starting with `prefix`.

    The ColumnTransformer selects by NAME and drops the rest, so the presence of an asset
    file does not say whether the booster needs a block; the booster cannot go stale, so
    ask it.

    Fail-open: any unexpected model layout returns True, rather than silently dropping a
    block the booster does need. Reads only OUR OWN artifact -- nothing test-derived,
    nothing pooled across test samples. Call it only after the LLM half has finished, so it
    cannot re-open the fork-after-threading hazard described in the NOTE in main().
    """
    try:
        _m = joblib.load(MODEL_PATH)
        for _b in getattr(_m, "bases", [_m]):
            for _n, _t, _cols in _b.named_steps["pre"].transformers:
                if _n == "dlg" and any(str(c).startswith(prefix) for c in _cols):
                    return True
        return False
    except Exception:
        return True


def _booster_cols():
    """Every `dlg` column the staged booster selects, or `[]` if the layout is unreadable.

    `_booster_selects` answers "does it use this prefix at all"; this answers "which exact
    names", which is what a block that emits a CHOSEN SUBSET of a wider vocabulary needs: the
    v2 spec chooses which v2 columns are built, so 'the booster wants v2 columns' is not enough
    to tell whether the right ones arrived. Fail-open (empty list) for the same reason
    `_booster_selects` fails open: an unreadable layout must not silently harden into an error.
    Reads only OUR OWN artifact -- nothing test-derived.
    """
    try:
        _m = joblib.load(MODEL_PATH)
        for _b in getattr(_m, "bases", [_m]):
            for _n, _t, _cols in _b.named_steps["pre"].transformers:
                if _n == "dlg":
                    return [str(c) for c in _cols]
        return []
    except Exception:
        return []



BASE_REPO = "Qwen/Qwen3-8B"
#: Where the runtime mounts every model listed in its `runtime/huggingface_models.txt`.
_HF_MOUNTS = ("/code_execution/huggingface_models", "huggingface_models")


def _resolve_base_ref():
    """(ref, mode) for the base Qwen3-8B weights — lifted from main_llmkt_lora.py.

    `mode` is "mount" (the documented runtime mechanism), "env" (local testing only), or
    "cache" — a bare repo id handed to transformers with `local_files_only=True`, which costs
    nothing if we are wrong about the mount and saves the run if the image caches models
    instead. It never touches the network.
    """
    import os  # noqa: PLC0415

    for mount in _HF_MOUNTS:
        cand = Path(mount) / BASE_REPO
        if (cand / "config.json").exists():
            return cand, "mount"
    env = os.environ.get("LLMKT_BASE_DIR", "")
    if env and (Path(env) / "config.json").exists():
        return Path(env), "env"
    return BASE_REPO, "cache"


def _load_base_plus_lora(base_ref, lora_dir, tok_dir, device: str):
    """Build the merged LLM: mounted base + our adapter, merged on CPU.

    The same loader as `main_llmkt_lora.py`, with the adapter and tokenizer folders passed in.

    🛑 The CPU merge is load-bearing: peft only upcasts the `B @ A` product to fp32 on CPU, so
    merging on the GPU silently moves the weights. Load bf16 on CPU, merge, THEN `.to(device)`.

    The tokenizer ships with the submission rather than coming from the mount, so prompts stay
    byte-identical no matter which revision of the base repo the runtime happens to mount.
    """
    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"      # last real token at attention_mask.sum()-1
    tokenizer.truncation_side = "left"    # keep the END
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(base_ref), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", local_files_only=True,
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            str(base_ref), torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )
    model = PeftModel.from_pretrained(model, str(lora_dir), local_files_only=True)
    model = model.merge_and_unload()      # CPU merge == the offline merge, bit for bit
    model = model.to(device)
    model.eval()
    return model, tokenizer


def step(name, fn):
    """Run fn(), printing only a static timing/pass-fail line -- never test
    content (competition rule: no logging of test content or aggregates). The
    rule is about test-derived content, not about prints per se. Every value in
    every line below is a package version, a shape/count of our OWN feature
    matrix, or an elapsed time -- nothing read from a transcript, LO text, or
    any test-row value."""
    t0 = time.time()
    try:
        info = fn()
        print(f"PROBE {name} OK {time.time() - t0:.1f}s {info or ''}")
        return info
    except Exception as e:  # noqa: BLE001
        print(f"PROBE {name} FAIL {type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc(limit=3)
        raise


def probe_versions():
    import sklearn
    import torch

    # Deliberately does NOT call torch.cuda.is_available() (or anything else
    # that touches torch.cuda) here: doing so initializes a CUDA context in
    # this main process, and vLLM's V1 engine later fork()s a worker
    # subprocess for its EngineCore -- forking a process that already has
    # CUDA initialized crashes with "Cannot re-initialize CUDA in forked
    # subprocess". Reading torch.__version__ alone never touches CUDA (lazy
    # init), so it's safe; version numbers only, no CUDA queries, until after
    # vLLM owns the process split.
    info = f"torch={torch.__version__} sklearn={sklearn.__version__}"
    try:
        import vllm
        info += f" vllm={vllm.__version__}"
    except ImportError:
        pass
    return info


def verify_manifest() -> None:
    """Re-hash every bundled asset and hard-fail on mismatch.

    A large submission.zip can get corrupted in transit (cluster -> laptop
    -> browser upload); a mangled weight file can otherwise load as silent
    garbage and produce inexplicable predictions instead of a clean crash.
    Runs first, before any model load, so a bad upload dies in the first
    minute of the smoke test rather than hour 5 of a full run.
    """
    import hashlib  # noqa: PLC0415

    if not MANIFEST_PATH.exists():
        return  # smaller model kinds may not ship a manifest
    n_files = 0
    for line in MANIFEST_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        h = hashlib.sha256()
        with open(ASSETS / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            print(f"PROBE manifest CORRUPT {rel}")
            sys.exit(f"CORRUPT ASSET: {rel} failed sha256 verification "
                      "(likely a bad upload) — aborting before any inference.")
        n_files += 1
    print(f"PROBE manifest OK {n_files} files verified")


def run_llmkt(return_proba: bool = False):
    """Merged Qwen dialogue-KT inference: per (session, LO) read the True/False
    logit and write probabilities. A frozen calibrator is applied only if one is
    bundled; otherwise the raw model probabilities are written. Each sample is
    scored independently; no test content is logged.

    With ``return_proba=True`` the calibrated Series is RETURNED instead of written,
    so the `ensemble` kind can blend it with the XGBoost model. The scoring path is otherwise
    identical, so the LLM half of the blend is the same computation as the LLM-only
    kind.

    If ``assets/llm_stub.txt`` is present, the model is not loaded and deterministic
    placeholder probabilities are produced instead (a plumbing test only; the submission
    does not ship that file)."""
    import os  # noqa: PLC0415
    # Feed the WHOLE transcript at an 8192-token budget.
    # llmkt_infer reads LLMKT_MAX_SEQ_LEN at import time, so set it BEFORE the import.
    os.environ.setdefault("LLMKT_MAX_SEQ_LEN", "8192")
    os.environ.setdefault("LLMKT_KEEP_HEADER", "1")  # header-preserving truncation

    import torch  # noqa: PLC0415

    from magnificat import llmkt_infer as li  # noqa: PLC0415

    feats = pd.read_csv(DATA_DIR / "test_features.csv", dtype=str)
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")
    _cal_path = ASSETS / "llmkt_calibrator.joblib"
    calibrator = joblib.load(_cal_path) if _cal_path.exists() else None

    # Verify the model weights survived upload. `unzip` prints 'bad CRC' on a
    # corrupted transfer but still writes the bytes, so safetensors loads GARBAGE
    # weights silently -> the model runs but predicts ~randomly.
    # Each shipped weight file is re-hashed against a sha256 manifest generated at pack
    # time (lora_sha256.txt for the adapter). These are our own files (not test data;
    # DQ-safe). On any mismatch, emit a recognizable base-rate submission + a loud
    # flag instead of a bogus score.
    if (ASSETS / "llm_stub.txt").exists():
        # deterministic per-response placeholder in [0.35, 0.90] from a hash of the
        # response_id -- our own identifier, never transcript content, and
        # computed per sample with no cross-sample pooling.
        import hashlib as _hl  # noqa: PLC0415
        print("LLM_STUB=ACTIVE ***NOT A REAL SUBMISSION*** placeholder probabilities")
        _p = pd.Series(
            [0.35 + 0.55 * (int(_hl.md5(str(r).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
             for r in feats["response_id"]],
            index=list(feats["response_id"]), dtype=float)
        if return_proba:
            return _p
        _o = submission_format.copy()
        _o["probability"] = _o["response_id"].map(_p).fillna(0.5).clip(EPS, 1 - EPS)
        _o.to_csv(SUBMISSION_PATH, index=False)
        print("inference complete")
        return None

    import hashlib  # noqa: PLC0415

    # --- LLM half: mounted base + our adapter, when the adapter was staged ---------------
    # The runtime MOUNTS every model in its huggingface_models.txt at
    # /code_execution/huggingface_models/<repo_id>/, and ships `peft`. So the LLM half is
    # a ~167 MB adapter merged onto the mounted base at load time. Checked for THIS model,
    # tensor by tensor, against the offline-merged weights (lmkt_mrft_sa_8b_all35k = LoRA
    # stage 2, the shipped adapter):
    #   base + lmkt_mrft_sa_8b_all35k  -> 399/399 bit-identical, max|diff| 0.000e+00
    # `--pt_model_name` only SEEDS the LoRA weights; it does not reparameterise them onto the
    # merged parent, so the adapter describes a delta on the base.
    #
    # 🛑 MERGE ON CPU. peft only upcasts the `B @ A` product to fp32 on CPU; a GPU-side merge
    # silently moves the weights. Load bf16 on CPU, merge_and_unload(), then .to(device) —
    # exactly what scripts/merge_llmkt_lora.py did offline.
    _lora_dir = ASSETS / "lora_llm"
    if _lora_dir.exists():
        _tok_dir = ASSETS / "tokenizer"
        _base_ref, _base_mode = _resolve_base_ref()
        print(f"BASE_MODEL mode={_base_mode}")
        # Integrity: the adapter is what WE ship, so hash it; the base is the runtime's, so
        # only the mechanism that resolved it (BASE_MODEL mode, above) is reported.
        _amf = ASSETS / "lora_sha256.txt"
        _bad = False
        for _line in (_amf.read_text().splitlines() if _amf.exists() else []):
            if not _line.strip():
                continue
            _hh, _nn = _line.split(None, 1)
            _p = _lora_dir / _nn.strip()
            _h = hashlib.sha256()
            with open(_p, "rb") as _f:
                for _chunk in iter(lambda: _f.read(1 << 20), b""):
                    _h.update(_chunk)
            _ok = _h.hexdigest() == _hh.strip()
            _bad = _bad or not _ok
            print(f"WEIGHT_CHECK {_nn.strip()} {'OK' if _ok else 'CORRUPT'}")
        if _bad:
            print("MODEL_INTEGRITY=CORRUPT the upload corrupted the adapter; predictions are "
                  "INVALID, please RE-UPLOAD (this score is not real)")
            _base = float(calibrator.transform([0.5])[0]) if calibrator is not None else 0.5
            _flat = max(EPS, min(1 - EPS, _base))
            if return_proba:
                return pd.Series(_flat, index=list(feats["response_id"]), dtype=float)
            _out = submission_format.copy()
            _out["probability"] = _flat
            _out.to_csv(SUBMISSION_PATH, index=False)
            return None
        print("MODEL_INTEGRITY=OK")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _preloaded = _load_base_plus_lora(_base_ref, _lora_dir, _tok_dir, _device)
    else:
        _preloaded = None

    _mdir = ASSETS / "qwen3_8b_full_merged"
    _manifest = _mdir / "weights_sha256.txt"
    _EXPECTED_SHARDS = {}
    if _preloaded is None:
        for _line in _manifest.read_text().splitlines():
            _line = _line.strip()
            if not _line:
                continue
            _hh, _nn = _line.split(None, 1)
            _EXPECTED_SHARDS[_nn.strip()] = _hh.strip()
    _corrupt = (_preloaded is None) and not _EXPECTED_SHARDS
    for _name, _exp in _EXPECTED_SHARDS.items():
        _h = hashlib.sha256()
        with open(_mdir / _name, "rb") as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b""):
                _h.update(_chunk)
        _ok = _h.hexdigest() == _exp
        _corrupt = _corrupt or not _ok
        print(f"WEIGHT_CHECK {_name} {'OK' if _ok else 'CORRUPT'}")
    if _corrupt:
        print("MODEL_INTEGRITY=CORRUPT the upload corrupted the model weights; "
              "predictions are INVALID, please RE-UPLOAD (this score is not real)")
        _base = float(calibrator.transform([0.5])[0]) if calibrator is not None else 0.5
        _flat = max(EPS, min(1 - EPS, _base))
        if return_proba:
            return pd.Series(_flat, index=list(feats["response_id"]), dtype=float)
        _out = submission_format.copy()
        _out["probability"] = _flat
        _out.to_csv(SUBMISSION_PATH, index=False)
        return None
    if _preloaded is None:
        print("MODEL_INTEGRITY=OK")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _preloaded is not None:
        model, tokenizer = _preloaded
    else:
        model, tokenizer = li.load_model(str(ASSETS / "qwen3_8b_full_merged"), device=device)
    true_token, false_token = li.get_true_false_tokens(tokenizer)

    # assemble each session's dialogue once (sessions are shared across responses)
    dia_cache: dict[str, object] = {}
    prompts, rids = [], []
    fallback_rids = []
    for _, r in feats.iterrows():
        sid = r["session_id"]
        if sid not in dia_cache:
            tx_path = DATA_DIR / "test_transcripts" / f"{sid}.csv"
            base = li.build_session_dialogue(pd.read_csv(tx_path)) if tx_path.exists() else None
            dia_cache[sid] = base
        base = dia_cache[sid]
        if not base:
            fallback_rids.append(r["response_id"])
            continue
        # Full transcript (max_turns=0 keeps every turn); llmkt_infer then
        # left-truncates to the 8192-token budget set above.
        dialogue = li.build_terminal_dialogue(base, max_turns=0)
        prompts.append(li.build_prompt(tokenizer, dialogue, str(r["learning_objective"]).strip()))
        rids.append(r["response_id"])

    raw = li.score_prompts(model, tokenizer, prompts, true_token, false_token,
                           batch_size=8, device=device)
    cal = calibrator.transform(raw) if calibrator is not None else raw
    proba = pd.Series(cal, index=rids)
    # responses with no usable transcript fall back to the base rate
    base_rate = (float(calibrator.transform([0.5])[0]) if calibrator is not None else 0.5) if fallback_rids else 0.5
    for rid in fallback_rids:
        proba.loc[rid] = base_rate

    if return_proba:
        return proba.reindex(list(feats["response_id"])).fillna(base_rate)

    out = submission_format.copy()
    out["probability"] = out["response_id"].map(proba).fillna(base_rate).clip(EPS, 1 - EPS)
    out.to_csv(SUBMISSION_PATH, index=False)
    print("inference complete")
    return None


def main() -> None:
    step("versions", probe_versions)
    verify_manifest()
    kind = KIND_PATH.read_text().strip() if KIND_PATH.exists() else "design"
    print(f"PROBE kind={kind}")

    # --- encoder pre-flight (loknn kinds only) ------------------------------------
    # `ensemble` builds no loknn columns and skips this. Loads the runtime-provided
    # `BAAI/bge-large-en-v1.5` encoder at startup; no test data is touched.
    if kind in ("ensemble_lk", *GBM_LK_KINDS):
        def _preflight_encoder():
            from magnificat import lo_knn as lk  # noqa: PLC0415
            path, is_local = lk.resolve_encoder()
            print(f"PROBE encoder_path mount={is_local} path={path}")
            if not is_local:
                print("PROBE encoder_path WARNING no runtime mount found; falling back to the "
                      "HF cache. In the competition runtime this WILL fail -- there is no cache "
                      "and no internet.")
            v = lk.encode(["preflight"])  # one constant string, never test content
            return f"dim={v.shape[1]}"
        step("encoder preflight", _preflight_encoder)

    if kind == "llmkt":
        run_llmkt()
        return

    llm_proba = None
    if kind in ENSEMBLE_KINDS:
        # LLM FIRST, deliberately: loading the LightGBM/XGBoost native lib spawns an
        # OpenMP pool, and a later fork() inherits its locks in a permanently-held
        # state (see the note below). Scoring the LLM before any booster import
        # keeps that ordering safe.
        llm_proba = run_llmkt(return_proba=True)
        print(f"PROBE llm_scored={llm_proba.notna().sum()}")

    feats = pd.read_csv(DATA_DIR / "test_features.csv", dtype=str)
    submission_format = pd.read_csv(DATA_DIR / "submission_format.csv")

    # NOTE: baseline.joblib (the XGBoost half) is loaded further down, right before
    # predict_proba(), not here. A LightGBM model loaded before a vLLM `LLM(...)` call (the
    # model_g_opt kind) makes vLLM's fork()-based EngineCore spawn hang (0% GPU util,
    # stuck in futex_do_wait, no traceback), almost certainly because the native library
    # starts an OpenMP thread pool at load time and a lock held at fork() time stays
    # locked in the child, where the thread that would release it does not exist.
    # Loading it late costs nothing.

    from magnificat import data as mdata  # noqa: PLC0415
    from magnificat import features  # noqa: PLC0415

    if kind == "meta":
        # metadata-only features
        X = features.build_meta_matrix(feats)
    elif kind == "meta_nc":
        # objective text/id + NC-curriculum match columns
        meta = features.build_meta_matrix(feats)
        curriculum = features.load_lo_curriculum()
        nc_df = features.load_nc_curriculum_df()
        X = features.add_lo_curriculum_features(meta, feats, curriculum, nc_df=nc_df)
    elif kind == "dialogue":
        # dialogue + objective-alignment columns, per session with no cross-sample pooling
        from magnificat import dialogue_features as dfeat  # noqa: PLC0415

        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")
        X = dfeat.build_dialogue_matrix(feats, store=store, show_progress=False)
    elif kind == "curriculum":
        # dialogue + structured + NC-curriculum columns
        from magnificat import dialogue_features as dfeat  # noqa: PLC0415

        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")
        dlg_X = dfeat.build_dialogue_matrix(feats, store=store, show_progress=False)
        design_X = features.build_design_matrix(feats, store=store, show_progress=False)

        t_cols = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
        available_t = [c for c in t_cols if c in design_X.columns]
        X = dlg_X.join(design_X[available_t])

        curriculum = features.load_lo_curriculum()
        nc_df = features.load_nc_curriculum_df()
        X = features.add_lo_curriculum_features(X, feats, curriculum, nc_df=nc_df)
    elif kind == "model_e":
        # adds objective-localised (LOC) columns from the train-frozen lo_idf.json map, never
        # refit on the test batch
        from magnificat import dialogue_features as dfeat  # noqa: PLC0415

        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")
        dlg_X = dfeat.build_dialogue_matrix(feats, store=store, show_progress=False)
        design_X = features.build_design_matrix(feats, store=store, show_progress=False)

        t_cols = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
        available_t = [c for c in t_cols if c in design_X.columns]
        X = dlg_X.join(design_X[available_t])

        curriculum = features.load_lo_curriculum()
        nc_df = features.load_nc_curriculum_df()
        X = features.add_lo_curriculum_features(X, feats, curriculum, nc_df=nc_df)

        idf, idf_default = dfeat.load_lo_idf()
        loc_X = dfeat.build_loc_matrix(feats, idf, idf_default, store=store, show_progress=False)
        X = X.join(loc_X)
        X = dfeat.add_loc_delta_features(X)
    elif kind in ("model_e2", "model_e3", *E3L_FEATURE_KINDS):
        # The shipped `ensemble` kind builds its XGBoost features in this branch. First the
        # 70 E2 columns, computed the SAME way the trainer's build_full_e_live does
        # (scripts/model_e2.py) so train and inference build identical columns:
        # dialogue + structured + NC-curriculum, then
        # magnificat.model_e2.build_e2_matrix applies the drop/twin/rate spec.
        from magnificat import dialogue_features as dfeat  # noqa: PLC0415
        from magnificat import model_e2 as e2spec          # noqa: PLC0415

        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")
        dlg_X = dfeat.build_dialogue_matrix(feats, store=store, show_progress=False)
        design_X = features.build_design_matrix(feats, store=store, show_progress=False)

        t_cols = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
        available_t = [c for c in t_cols if c in design_X.columns]
        missing_t = [c for c in t_cols if c not in available_t]
        if missing_t:
            raise RuntimeError(
                f"model_e2 inference missing {len(missing_t)}/{len(t_cols)} structured "
                f"cols vs training (STRUCTURED_COLS drift, not a cache issue): {missing_t}")
        X = dlg_X.join(design_X[available_t])

        curriculum = features.load_lo_curriculum()
        nc_df = features.load_nc_curriculum_df()
        X = features.add_lo_curriculum_features(X, feats, curriculum, nc_df=nc_df)

        full_cols = (list(dfeat.DIALOGUE_COLS) + list(dfeat.LO_COLS) + available_t
                     + list(features.LO_CURRICULUM_COLS))
        X, _ = e2spec.build_e2_matrix(X, full_cols)

        # --- the 31 evidence columns ------------------------------------------------
        # The distilled answer log, contingency and error-repair blocks
        # (magnificat.evidence_features), appended to the E2 columns. The kind marker, not
        # file presence, is the switch.
        if kind in ("model_e3", *E3L_FEATURE_KINDS):
            from magnificat import evidence_features as evf  # noqa: PLC0415

            clf_path = ASSETS / "turn_correctness_clf.joblib"
            if not clf_path.exists():
                raise RuntimeError(
                    f"model_e3 needs the distilled per-turn correctness annotator at "
                    f"{clf_path.name}; stage it with "
                    f"`build_submission.py --kind model_e3`.")
            ev = evf.build_evidence_matrix(feats, store=store, clf=joblib.load(clf_path))
            X = X.join(ev)
            missing_e = [c for c in evf.EVIDENCE_COLS if c not in X.columns]
            if missing_e:
                raise RuntimeError(
                    f"model_e3 inference missing {len(missing_e)} evidence cols vs "
                    f"training: {missing_e}")

        # --- the 21 distilled NTO move rates (E3L feature kinds only) ------------
        # Gated on the KIND, not on file presence. The submission's booster does not select
        # these columns (the pipeline selects by NAME), so for it they are built but unused.
        if kind in E3L_FEATURE_KINDS:
            from magnificat import nto_features as ntof  # noqa: PLC0415

            nto_path = ASSETS / "nto_moves_clf.joblib"
            if not nto_path.exists():
                raise RuntimeError(
                    f"ensemble needs the distilled move annotator at {nto_path.name}; "
                    f"stage it with `build_submission.py --kind ensemble`.")
            nt = ntof.build_nto_matrix(feats, store=store, clf=joblib.load(nto_path))
            X = X.join(nt)
            missing_n = [c for c in ntof.NTODST_COLS if c not in X.columns]
            if missing_n:
                raise RuntimeError(
                    f"ensemble inference missing {len(missing_n)} ntodst cols vs "
                    f"training: {missing_n}")

        # --- the 12 distilled student-state rates, and the 34 interview-codebook columns --
        # Gated on ASSET PRESENCE rather than KIND, unlike the blocks above. The pipeline
        # selects by NAME, so joining columns the booster does not select is inert. The
        # shipped booster was trained with `--swap-student`, which swaps E2's 14 student
        # regexes out of the booster's column LIST while leaving them in the matrix. The
        # submission stages both classifiers.
        #
        # Both twins are parity-verified against the training CSVs at
        # max|Δ| = 0.000000 (scripts/check_studst_parity.py, scripts/check_doc29_parity.py);
        # re-run those gates after ANY change to the truncation or position_frac logic.
        if kind in E3L_FEATURE_KINDS:
            stu_path = ASSETS / "stu_states_clf.joblib"
            if stu_path.exists():
                from magnificat import student_features as stuf  # noqa: PLC0415

                X = X.join(stuf.build_studst_matrix(
                    feats, store=store, clf=joblib.load(stu_path)))
                missing_s = [c for c in stuf.STUDST_COLS if c not in X.columns]
                if missing_s:
                    raise RuntimeError(
                        f"inference missing {len(missing_s)} studst cols vs training: "
                        f"{missing_s}")

            d29_path = ASSETS / "doc29_clf.joblib"
            if d29_path.exists():
                from magnificat import doc29_features as d29f  # noqa: PLC0415

                X = X.join(d29f.build_d29_matrix(
                    feats, store=store, clf=joblib.load(d29_path)))
                missing_d = [c for c in d29f.D29DST_COLS if c not in X.columns]
                if missing_d:
                    raise RuntimeError(
                        f"inference missing {len(missing_d)} d29dst cols vs training: "
                        f"{missing_d}")

        # --- the taxonomy-v2 move block (a narrow shape, or a family such as v2ht_/v2hs_) ---
        # Presence-gated like studst_/d29dst_ above. `assets/v2_spec.txt` says which shape the
        # booster was trained on, so inference builds that shape rather than this file's
        # defaults. The submission's spec is `narrow=none`, `source=shrunk`: the shrunk-rate
        # family (`v2ht_`/`v2hs_`) built by `build_v2_family_matrix`.
        #
        # Twin parity is checked end to end by `scripts/check_v2_family_parity.py`.
        # Nothing test-derived is printed.
        v2t_path, v2s_path = ASSETS / "v2t_clf.joblib", ASSETS / "v2s_clf.joblib"
        spec_path = ASSETS / "v2_spec.txt"
        if (kind in E3L_FEATURE_KINDS and v2t_path.exists() and v2s_path.exists()
                and spec_path.exists()):
            from magnificat import v2_features as v2f  # noqa: PLC0415

            _t0 = time.time()
            _spec = dict(
                ln.split("=", 1) for ln in spec_path.read_text().split("\n")
                if "=" in ln and not ln.strip().startswith("#"))
            _which = _spec.get("narrow", "narrows").strip()
            _source = _spec.get("source", "rate").strip()
            _fams = [f for f in _spec.get("families", "").strip().split(",") if f]
            _tc, _sc = joblib.load(v2t_path), joblib.load(v2s_path)
            if _which == "none":
                v2 = v2f.build_v2_family_matrix(feats, store=store, tclf=_tc, sclf=_sc,
                                                families=[_source] + _fams)
            else:
                v2 = v2f.build_v2_narrow_matrix(feats, store=store, tclf=_tc, sclf=_sc,
                                                which=_which, source=_source,
                                                extra_families=_fams)
            X = X.join(v2)
            print(f"PROBE v2moves OK {time.time() - _t0:.1f}s cols={v2.shape[1]} "
                  f"spec={_which}/{_source}" + (f"+{'+'.join(_fams)}" if _fams else ""),
                  flush=True)
            _sel = [c for c in _booster_cols() if c.startswith(
                ("v2t_", "v2s_", "v2n", "v2m", "v2k", "v2u", "v2b_", "v2sl_", "v2d_",
                 "v2x_", "v2r_", "v2q_", "v2w_"))]
            _miss = [c for c in _sel if c not in X.columns]
            if _miss:
                raise RuntimeError(
                    f"the booster selects {len(_sel)} v2 columns and inference produced "
                    f"{len(_sel) - len(_miss)}; missing: {_miss[:8]}")


        # --- the 3 objective-neighbourhood columns (ensemble_lk / model_e3l only) ----
        # No cross-test pooling: every column is a
        # function of ONE response's objective text against a frozen table of training
        # objectives; nothing is fitted, normalised or aggregated over the test batch.
        if kind in ("ensemble_lk", *GBM_LK_KINDS):
            from magnificat import lo_knn as lk  # noqa: PLC0415

            tab_path = ASSETS / "loknn_table.joblib"
            if not tab_path.exists():
                raise RuntimeError(
                    f"ensemble_lk needs the frozen neighbourhood table at "
                    f"{tab_path.name}; stage it with "
                    f"`build_submission.py --kind ensemble_lk`.")
            table = joblib.load(tab_path)
            uniq = feats.drop_duplicates("learning_objective_id")
            uids = uniq["learning_objective_id"].to_numpy()
            vecs = lk.encode(uniq["learning_objective"].fillna("").tolist())
            Fu = lk.transform(uids, {l: vecs[i] for i, l in enumerate(uids)}, table)
            Fu.index = uids
            # map objective -> response by id, never by row position
            lo_by_rid = feats.set_index("response_id")["learning_objective_id"]
            F = Fu.reindex(lo_by_rid.reindex(X.index).to_numpy())
            F.index = X.index
            if F.isna().any().any():
                raise RuntimeError("loknn produced NaN; transform must fall back to "
                                   "the training prior, never NaN")
            X = X.join(F)
            missing_k = [c for c in lk.LOKNN_COLS if c not in X.columns]
            if missing_k:
                raise RuntimeError(
                    f"ensemble_lk inference missing {missing_k} vs training")
            print(f"PROBE loknn table_objectives={len(table.lo_ids)}")
    elif kind == "model_g_opt":
        # live-LLM feature kind (needs Qwen2.5-7B-Instruct in assets/)
        from magnificat import dialogue_features as dfeat  # noqa: PLC0415
        from magnificat import sandpiper_live_infer as sli  # noqa: PLC0415

        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")

        _t0 = time.time()
        dlg_X = dfeat.build_dialogue_matrix(feats, store=store, show_progress=False)
        design_X = features.build_design_matrix(feats, store=store, show_progress=False)

        t_cols = [c for c in features.STRUCTURED_COLS if c not in features.LO_DIFFICULTY_COLS]
        available_t = [c for c in t_cols if c in design_X.columns]
        X = dlg_X.join(design_X[available_t])

        curriculum = features.load_lo_curriculum()
        nc_df = features.load_nc_curriculum_df()
        X = features.add_lo_curriculum_features(X, feats, curriculum, nc_df=nc_df)

        idf, idf_default = dfeat.load_lo_idf()
        loc_X = dfeat.build_loc_matrix(feats, idf, idf_default, store=store, show_progress=False)
        X = X.join(loc_X)
        X = dfeat.add_loc_delta_features(X)
        print(f"PROBE classical_features OK {time.time() - _t0:.1f}s shape={X.shape}")

        qwen_dir = ASSETS / "qwen2.5-7b-instruct"
        encoder_dir = ASSETS / "all-MiniLM-L6-v2"
        pca_path = ASSETS / "sandpiper_misconception_pca.pkl"

        _t1 = time.time()
        try:
            llm_X = sli.annotate_and_build_gopt_llm_matrix(
                feats, store, str(qwen_dir), str(encoder_dir),
                misconception_pca_path=pca_path if pca_path.exists() else None,
                groups=frozenset({"student", "tutor_whole", "rubric"}),
            )
        except Exception as e:  # noqa: BLE001
            print(f"PROBE gopt_llm_features FAIL {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc(limit=3)
            raise
        print(f"PROBE gopt_llm_features OK {time.time() - _t1:.1f}s shape={llm_X.shape}")
        X = X.join(llm_X)
    else:
        store = mdata.TranscriptStore(DATA_DIR / "test_transcripts")
        X = features.build_design_matrix(
            feats[["response_id", "session_id", "learning_objective"]],
            store=store,
            show_progress=False,
        )

    # The booster is loaded here, after the LLM half and any vLLM use above, not at the
    # top of main(); see the NOTE after the test files are read.
    _t2 = time.time()
    model = joblib.load(MODEL_PATH)
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)

    # --- ensemble: fixed-weight probability blend with the LLM --------------------
    # p = (1-w)*p_LLM + w*p_XGBoost, w frozen at TRAIN time (assets/blend_weight.txt, 0.4300
    # in the submission) and never fitted here -- test predictions are combined per
    # sample, with no pooling across test rows. Probability space, not a learned logit
    # stack.
    if kind in ENSEMBLE_KINDS:
        w = float((ASSETS / "blend_weight.txt").read_text().strip())
        if not 0.0 <= w <= 1.0:
            raise RuntimeError(f"blend weight {w} outside [0,1]")
        gbm = proba.reindex(list(feats["response_id"]))
        llm = llm_proba.reindex(list(feats["response_id"]))
        n_bad = int(gbm.isna().sum() + llm.isna().sum())
        gbm = gbm.fillna(0.5)
        llm = llm.fillna(0.5)
        proba = (1.0 - w) * llm + w * gbm
        print(f"PROBE blend w={w:.2f} missing={n_bad}")
    print(f"PROBE predict OK {time.time() - _t2:.1f}s shape={proba.shape}")

    out = submission_format.copy()
    out["probability"] = (
        out["response_id"].map(proba).fillna(0.5).clip(EPS, 1 - EPS)
    )
    out.to_csv(SUBMISSION_PATH, index=False)
    # Per the rules, do NOT log anything derived from the test data (counts,
    # sums, means, transcript/objective text) -- shapes/timings above are
    # about our own feature matrices, never row content.
    print("inference complete")


if __name__ == "__main__":
    main()
