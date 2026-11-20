#!/usr/bin/env python
r"""Assemble a submission zip and optionally dry-run it on a sample of training sessions.

Layout: main.py, assets/ (from the fit_final_v2.py staging) and assets/ace/ (the modules main.py
imports). The dry run extracts the zip into a scratch dir, builds a test-format data/ from
training sessions, runs main.py as the runtime would, checks the output format and scores it
against the known labels.

    python -m ace.submission.build_zip --staging outputs/staging --zip outputs/submission.zip \
        --dry-run 80 --data-dir data --transcripts-dir data/train_transcripts
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ACE_MODULES = ["__init__.py", "features.py", "transcripts.py", "utils.py",
               "tabular_v2_features.py",  # structured transcript features (tabular_v2)
               "llm/__init__.py", "llm/serialize.py", "llm/score_worker.py",
               # bundle metadata (optional bundle.json) drives engine construction;
               # model_family supplies the family names bundle.py uses, and runtime_check
               # is imported by bundle.check_engine_supports() on every engine build, so it
               # is a runtime dependency of score_worker. None imports vllm/torch at load.
               "llm/bundle.py", "llm/model_family.py", "llm/runtime_check.py",
               # a reader may declare prompt_mode=objective_chunks, which needs the BGE
               # context builder and the prompt assembler at test time; both are inert
               # for standard-prompt readers.
               "llm/objective_context.py", "llm/prepare_objective_contexts.py"]
STORED_SUFFIXES = {".safetensors"}  # model weights don't deflate: store them


def _write(z: zipfile.ZipFile, src: Path, arcname: str) -> None:
    ct = zipfile.ZIP_STORED if src.suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
    z.write(src, arcname, compress_type=ct)


def build(staging: Path, zip_path: Path, llm_bundles: dict[str, Path] | None = None,
          llm_stack: Path | None = None) -> None:
    """llm_bundles maps an assets subdir to a stage_llm.sh bundle (base/ + adapters/):
    {"llm": p} for a single reader, {"llm_4b": p4, "llm_8b": p8} for several.
    """
    src_ace = Path(__file__).resolve().parents[1]  # src/ace
    main_py = Path(__file__).resolve().parent / "main.py"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name and rename only after verification, so an interrupted
    # build can never be mistaken for a submittable zip.
    tmp_path = zip_path.with_suffix(".zip.part")
    print(f"[zip] writing {tmp_path} ...")
    # A duplicate arcname is not a benign zip quirk: python's zipfile silently reads the
    # last copy, but the platform unpacks with `unzip`, which prompts to overwrite, gets EOF
    # on stdin and aborts the run before main.py executes. Track every name and fail the
    # build instead (e.g. a stack json both in staging/assets and passed via --llm-stack).
    written: set[str] = set()

    def _put(src: Path, arc: str, stored: bool = True) -> None:
        if arc in written:
            raise SystemExit(
                f"[zip] duplicate archive entry: {arc}\n"
                f"      (a staging file collides with an explicitly passed asset -- "
                f"remove the copy under {staging}/assets/)")
        written.add(arc)
        if stored:
            _write(z, src, arc)
        else:
            z.write(src, arc)

    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        _put(main_py, "main.py", stored=False)
        for m in ACE_MODULES:
            _put(src_ace / m, f"assets/ace/{m}", stored=False)
        assets = staging / "assets"
        stack_arc = "assets/llm_stack.json"
        for f in sorted(assets.rglob("*")):
            if f.is_file():
                arc = f"assets/{f.relative_to(assets)}"
                if llm_bundles and arc == stack_arc:
                    # the explicitly passed --llm-stack is authoritative; skip the copy
                    print(f"[zip] skipping staged {arc} (superseded by --llm-stack)")
                    continue
                _put(f, arc)
        if llm_bundles:  # each bundle base/ + adapters/*/ -> assets/<subdir>/
            for subdir, bpath in llm_bundles.items():
                for f in sorted(bpath.rglob("*")):
                    if f.is_file():
                        _put(f, f"assets/{subdir}/{f.relative_to(bpath)}")
            _put(llm_stack, stack_arc, stored=False)
    print("[zip] verifying archive integrity ...")
    with zipfile.ZipFile(tmp_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"[zip] CRC check failed on {bad}")
        names = z.namelist()
        dupes = {x for x in names if names.count(x) > 1}
        if dupes:
            raise SystemExit(f"[zip] duplicate entries in archive: {sorted(dupes)}")
        n = len(names)
    tmp_path.replace(zip_path)
    print(f"[zip] done: {zip_path} — {n} files, {zip_path.stat().st_size/1e6:.0f} MB")


def dry_run(zip_path: Path, run_dir: Path, data_dir: Path, tdir: Path, n_sessions: int) -> None:
    from ace.utils import logloss

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    print(f"[dry] extracting zip to {run_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(run_dir)

    feats = pd.read_csv(data_dir / "train_features.csv")
    labels = pd.read_csv(data_dir / "train_labels.csv")
    rng = np.random.default_rng(0)
    # sample sessions, biased to include multi-response ones
    counts = feats.groupby("session_id").size()
    multi = counts[counts > 1].index.to_numpy()
    single = counts[counts == 1].index.to_numpy()
    pick = np.concatenate([rng.choice(multi, n_sessions // 2, replace=False),
                           rng.choice(single, n_sessions - n_sessions // 2, replace=False)])
    sample = feats[feats["session_id"].isin(pick)].reset_index(drop=True)

    # simulate unseen objectives (the test set's regime): remap half the sampled ids to
    # fake ones, keeping the objective text, so the runtime kNN fallback is exercised and
    # not just the known-id table lookup
    ids = sample["learning_objective_id"].unique()
    mangle = set(np.random.default_rng(1).choice(ids, len(ids) // 2, replace=False))
    sample["learning_objective_id"] = [f"unseen-{i}" if i in mangle else i
                                       for i in sample["learning_objective_id"]]
    print(f"[dry] remapped {len(mangle)}/{len(ids)} objective ids to unseen")

    ddir = run_dir / "data"
    (ddir / "test_transcripts").mkdir(parents=True)
    sample.to_csv(ddir / "test_features.csv", index=False)
    pd.DataFrame({"response_id": sample["response_id"], "probability": 0.5}).to_csv(
        ddir / "submission_format.csv", index=False)
    for s in pick:
        shutil.copy2(tdir / f"{s}.csv", ddir / "test_transcripts" / f"{s}.csv")

    print(f"[dry] running main.py on {len(sample)} rows / {n_sessions} sessions ...")
    # Some local setups cannot run vLLM: the LLM stage then falls back to tabular here;
    # the LLM path itself is validated in the official container.
    r = subprocess.run([sys.executable, "main.py"], cwd=run_dir,
                       capture_output=True, text=True, timeout=3600)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit(f"[dry] main.py FAILED (exit {r.returncode})")

    out = pd.read_csv(run_dir / "submission.csv")
    assert list(out.columns) == ["response_id", "probability"], "bad columns"
    assert len(out) == len(sample), "row count mismatch"
    assert out["probability"].between(0, 1).all(), "probabilities out of range"
    assert not out["probability"].isna().any(), "NaN probabilities"
    scored = out.merge(labels, on="response_id", validate="1:1")
    ll = logloss(scored["is_correct"], scored["probability"])
    print(f"[dry] OK — format valid; sample log loss {ll:.4f} "
          f"(sessions seen in training, so expect below the ~0.53 OOF)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", default="outputs/submissions/staging")
    ap.add_argument("--zip", default="outputs/submissions/submission_blend_v1.zip")
    ap.add_argument("--llm-bundle", default=None,
                    help="single-LLM: stage_llm.sh dir (base/ + adapters/) -> assets/llm/")
    ap.add_argument("--llm-bundle-multi", nargs="+", default=None, metavar="SUBDIR=PATH",
                    help="multi-LLM: repeated subdir=bundle pairs -> assets/<subdir>/ "
                         "(e.g. llm_4b=outputs/submissions/llm_bundle_4b "
                         "llm_8b=outputs/submissions/llm_bundle_8b); the stack json's "
                         "'bundles' must reference the same subdirs")
    ap.add_argument("--llm-stack", default="outputs/models/sub6_fit/llm_stack.json",
                    help="stack coefficients json (required with any --llm-bundle*)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N_SESSIONS")
    ap.add_argument("--data-dir")
    ap.add_argument("--transcripts-dir")
    args = ap.parse_args()

    staging = Path(args.staging).expanduser()
    zip_path = Path(args.zip).expanduser()
    # resolve bundles: single (--llm-bundle -> assets/llm/) or multi (subdir=path)
    bundles: dict[str, Path] = {}
    if args.llm_bundle:
        bundles["llm"] = Path(args.llm_bundle).expanduser()
    for spec in (args.llm_bundle_multi or []):
        if "=" not in spec:
            raise SystemExit(f"[zip] --llm-bundle-multi entry must be SUBDIR=PATH: {spec}")
        subdir, pth = spec.split("=", 1)
        bundles[subdir] = Path(pth).expanduser()
    if bundles:
        stack = Path(args.llm_stack).expanduser()
        # each bundle: base/ + adapters/<name>/ (fold0..4 mini-ensemble, or a single
        # full/ full-fit at k=1). main.py adapts k to however many adapters are present.
        # A dense bundle (bundle.json adapter_mode=dense) legitimately has no adapters/,
        # because the LoRA deltas are already merged into base/; validate_bundle
        # enforces the right shape for whichever mode is declared.
        # base_mode=external ships no local base/: the runtime supplies the model
        # from huggingface_models/{org}/{model}/. Requiring base/ there would reject a
        # valid bundle at build time; its metadata is validated instead, and the
        # architecture check runs in the worker where the path actually exists.
        from ace.llm.bundle import (BASE_MODE_EMBEDDED, BundleValidationError,
                                    load_bundle_meta, validate_bundle)
        for subdir, bpath in bundles.items():
            try:
                bmeta = load_bundle_meta(bpath)
                embedded = bmeta.base_mode == BASE_MODE_EMBEDDED
                if embedded and not (bpath / "base").is_dir():
                    raise SystemExit(f"[zip] bundle {subdir}={bpath} lacks base/")
                meta = validate_bundle(bpath, bmeta, resolve_base=embedded)
            except BundleValidationError as e:
                raise SystemExit(f"[zip] bundle {subdir}={bpath} is not deployable:\n{e}")
            if not embedded:
                print(f"[zip] NOTE {subdir}: base_mode={meta.base_mode} — no weights are "
                      f"bundled; the runtime must provide {meta.external_base!r}. "
                      f"Its architecture is validated at load time, not here.")
            if meta.incomplete_cv:
                # Loud but not fatal: an incomplete-CV reader is a legitimate exploratory
                # probe, as long as nobody calls it validated.
                print(f"[zip] NOTE {subdir}: INCOMPLETE CV — folds {meta.folds}, "
                      f"missing {meta.missing_folds}. This bundle must not be described "
                      f"as a cross-validated reader.")
        if not stack.is_file():
            raise SystemExit(f"[zip] --llm-stack {stack} not found")
        build(staging, zip_path, bundles, stack)
    else:
        build(staging, zip_path)
    if args.dry_run:
        if not (args.data_dir and args.transcripts_dir):
            raise SystemExit("--dry-run needs --data-dir and --transcripts-dir")
        dry_run(zip_path, zip_path.parent / "dryrun",
                Path(args.data_dir).expanduser(),
                Path(args.transcripts_dir).expanduser(), args.dry_run)


if __name__ == "__main__":
    main()