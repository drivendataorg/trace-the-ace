#!/usr/bin/env python
"""Inference entry point: score (session, learning objective) pairs with the trained model.

``models/submission/main.py`` is the exact program the competition platform ran. It reads a fixed
layout (``./data/test_features.csv``, ``./data/submission_format.csv``,
``./data/test_transcripts/<session_id>.csv``) and writes ``./submission.csv``. This wrapper checks
your inputs, builds that layout in a temporary directory around them, runs ``main.py`` unchanged,
and copies the result out, so every prediction goes through byte for byte the scored inference
code.

    # the competition test set, as downloaded into data/raw/  ->  data/processed/submission.csv
    python src/run_inference.py

    # preflight only: inputs, package versions, GPU, base model, asset checksums; no model is loaded
    python src/run_inference.py --check

    # the same preflight on the shipped one-row example, without a GPU
    python src/run_inference.py --check --cpu --features data/examples/new_data/rows.csv \\
        --transcripts data/examples/new_data/transcripts

    # any new data: a features CSV and a folder (or zip) of transcripts
    python src/run_inference.py --features my_rows.csv --transcripts my_transcripts/ --out preds.csv

    # no GPU (a few rows only: a full-length transcript takes minutes per row on CPU)
    python src/run_inference.py --cpu --features my_rows.csv --transcripts my_transcripts/

Flags
-----
``--features``     CSV with columns response_id, session_id, learning_objective_id,
                   learning_objective. An objective id never seen in training is expected and
                   gets no adjustment: the model reads the objective's text. A reused training
                   id shifts the XGBoost half slightly (only 6 of the 398 training ids are ever
                   split on, about 0.2% of the split gain). Use one id per objective text: the
                   curriculum features lo_year_group, lo_ks_level and lo_nc_sim are computed once
                   per id, from the first text seen with it. Default: ``data/raw/test_features*.csv``.
``--transcripts``  folder of ``<session_id>.csv`` files with columns session_id, utterance_id,
                   role, content, timestamp (role tutor/student/background; timestamp HH:MM:SS
                   elapsed), or a zip of them. Default: ``data/raw/test_transcripts/`` if that
                   folder holds ``*.csv`` files directly inside it, otherwise
                   ``data/raw/test_transcripts*.zip``.
``--submission-format``  CSV whose response_id column sets which rows are written, and in which
                   order. Default without ``--features``: picked from
                   ``data/raw/submission_format*.csv`` (see below). Default with ``--features``:
                   the row order of ``--features``.
``--out``          where to write the predictions (a file path, not a folder; checked before the
                   run starts). Default: ``data/processed/submission.csv``.
``--submission-dir``  folder holding ``main.py`` and ``assets/``. Default: ``models/submission``
                   (the scored submission, with its seven text-classifier files re-saved so that
                   their vocabularies are stored in sorted order and comments edited; predictions
                   are identical).
                   Point it at a rebuilt tree (for example
                   ``build/workspace/build/submission``) to run that through the same staging.
``--check``        run the preflight below and stop: format selection, input checks, package
                   versions, device, base-model location and the asset manifest. Loads no model.
                   Ends with ``CHECK PASSED``, or ``CHECK PASSED WITH <n> WARNING(S)`` when any
                   WARNING line was printed.
``--cpu``          allow running when no GPU is visible (main.py then runs on CPU). Without it the
                   wrapper stops, because a CPU run of the full test set would take days. It does
                   not hide a GPU that is there; to force CPU on a GPU machine also set
                   ``CUDA_VISIBLE_DEVICES=""``.
``--skip-version-check``  do not compare installed package versions with ``requirements.txt``.
``--allow-env-override``  pass your own ``LLMKT_MAX_SEQ_LEN`` and ``LLMKT_KEEP_HEADER`` to
                   main.py. Without it the wrapper sets them to the scored values (8192 and 1)
                   and prints a note if your environment held other values.

What the wrapper checks before it starts main.py
------------------------------------------------
* **Format file.** DrivenData serves two format files: ``submission_format_ZQLcKx7.csv``
  (10,508 rows, the released test set) and ``submission_format_5muR4s3.csv`` (100 rows, the
  platform's smoke test; none of its ids are in the released test features). Of the
  ``data/raw/submission_format*.csv`` files, the wrapper keeps those whose response_ids are all in
  the features and takes the one with the most rows. It stops if none qualifies, and it applies
  the same check to a file passed with ``--submission-format``.
* **Package versions.** scikit-learn and xgboost must match ``requirements.txt`` exactly: the
  XGBoost model and the distilled classifiers are pickled objects, and with another version
  main.py fails only after the hours-long LLM pass, with an AttributeError and no warning
  (main.py silences all warnings). Other pinned packages that differ only print a warning.
* **Device.** Prints the torch build, its CUDA version and the GPU; stops if no GPU is visible
  unless ``--cpu`` is given. Sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` unless you
  set it yourself: it lowers the peak reserved GPU memory at batch 8 (to 32.6 GiB) and leaves
  the predictions bit-identical. Warns, when a GPU is visible, if the ``flash_attn`` package is
  installed: main.py then asks transformers for ``flash_attention_2`` first. flash-attn is
  neither in ``requirements.txt`` nor in the competition runtime's lock file at commit ea9a817,
  and every run in ``reports/gpu_noise/`` used PyTorch SDPA. How much flash-attn moves the
  predictions has not been measured. On CPU transformers cannot use flash-attn, and main.py
  falls back to SDPA.
* **Scored settings.** Sets ``LLMKT_MAX_SEQ_LEN=8192`` and ``LLMKT_KEEP_HEADER=1`` for main.py,
  the values of the scored run, even if your environment holds other values (a note says so).
  Pass ``--allow-env-override`` to keep yours; the run then warns that its predictions may
  differ from the scored setting.
* **Base model.** Looks for Qwen/Qwen3-8B where main.py does, in its order: the platform mount
  ``/code_execution/huggingface_models/Qwen/Qwen3-8B``, ``./huggingface_models/Qwen/Qwen3-8B``
  (relative to main.py's working directory, which is the temporary directory, so never found
  here), ``LLMKT_BASE_DIR`` (set to ``models/huggingface_models/Qwen/Qwen3-8B`` when you have
  not set it and that folder holds the model), and finally the local Hugging Face cache, read
  offline. Relative ``LLMKT_BASE_DIR``, ``HF_HOME``, ``HF_HUB_CACHE`` and ``TRANSFORMERS_CACHE``
  values are made absolute first, because main.py runs in the temporary directory. Stops if the
  model is in none of these places, or if the cached copy lacks a weight file. The cache needs
  ``refs/main`` next to the snapshot: a download pinned to a commit hash with huggingface_hub
  0.36.2 does not write it, and without it main.py's offline load fails with an OSError. Warns
  if the cached snapshot is not revision ``b968826d9c46dd6066d109eabc6255188de91218``, the
  base the adapter was trained and scored on.
* **response_id.** Must be non-empty and unique. main.py keys rows by response_id and reads the
  format file without ``dtype=str``, so ids such as ``1`` or ``007`` would come back as integers,
  match nothing and give 0.5 on every row. When an id would not survive that, the wrapper replaces
  every id with ``rid_<n>`` (in input order) for the run and maps the original strings back in the
  output, so ``007`` stays ``007``.
* **Other fields.** session_id, learning_objective_id and learning_objective must be non-empty
  after main.py's own CSV parsing, which turns ``NA``, ``None``, ``null``, ``nan``, ``n/a`` and
  similar cells into missing values. An objective whose literal text is one of those words
  therefore cannot be scored by the unchanged program.
* **Transcripts.** Only the transcripts of the requested sessions are copied into the temporary
  directory (your files are never modified). Roles are stripped and lower-cased in the copy. A
  0-byte file counts as missing. A transcript with utterances but no ``tutor`` line stops the run
  before the LLM pass, because main.py's feature code fails on it after the LLM pass. A missing or
  header-only transcript is allowed: for that row the LLM half falls back to its Platt map at 0.5
  (≈ 0.61), and the XGBoost half still scores it with empty-transcript features. The run stops,
  though, when none of the requested sessions has a transcript file (usually a wrong path or a
  folder unpacked one level too deep), and, with the default inputs, when
  ``data/raw/test_transcripts/`` lacks some sessions while a ``test_transcripts*.zip`` sits next to
  it (a partly unpacked folder: delete it, and the zip is used).
* **Transcript format.** Warns, without stopping, on transcripts whose ``utterance_id`` values
  are not all whole numbers or repeat, and on ``timestamp`` values that are not ``HH:MM:SS``.
  main.py orders the utterances by ``utterance_id``: the XGBoost half reads it as a number (a value
  that is not a number, an empty cell included, goes to the end), the LLM half with pandas'
  defaults (one text value makes the whole column text, so ``10`` sorts before ``2``), and the
  order of repeated ids is not defined. The XGBoost half's timing features read a timestamp as
  missing unless it is three colon-separated whole numbers. A note also counts utterances whose
  whole content is a word pandas reads as missing (``NA``, ``None``, ``null``, ...): main.py
  reads them as empty, as it did in the scored run, so the LLM prompt leaves them out.
* **Output path.** ``--out`` must not be an existing folder, and its folder must be creatable and
  writable. If writing it still fails after main.py has finished, the predictions are saved to a
  fallback file whose path is printed, so the run is not lost.

Calling ``models/submission/main.py`` directly (the competition-container route) does none of
these checks.

A single new data point is a one-row features CSV plus one transcript file. Rows are scored
independently; nothing is pooled across rows. Needs the base model Qwen/Qwen3-8B in
``models/huggingface_models/Qwen/Qwen3-8B`` (README, Setup), at ``LLMKT_BASE_DIR`` or in the
local Hugging Face cache with ``refs/main`` set (see "Base model" above).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
REQUIREMENTS = REPO / "requirements.txt"
LOCAL_BASE = REPO / "models" / "huggingface_models" / "Qwen" / "Qwen3-8B"
PLATFORM_BASE = Path("/code_execution/huggingface_models/Qwen/Qwen3-8B")  # main.py's first choice
BASE_REPO = "Qwen/Qwen3-8B"
BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"   # the base the adapter was trained on
#: main.py setdefault()s these; the wrapper pins them so a stray shell variable cannot change them
SCORED_ENV = {"LLMKT_MAX_SEQ_LEN": "8192", "LLMKT_KEEP_HEADER": "1"}
TIMESTAMP_RE = r"\d+:\d{2}:\d{2}"
REQUIRED = ["response_id", "session_id", "learning_objective_id", "learning_objective"]
TRANSCRIPT_REQUIRED = ["utterance_id", "role", "content", "timestamp"]
KNOWN_ROLES = {"tutor", "student", "background"}
#: pickled with these packages: a version mismatch fails only after the LLM pass
HARD_PINS = {"scikit-learn", "xgboost"}
SMOKE_NOTE = ("DrivenData serves two format files: submission_format_ZQLcKx7.csv (10,508 rows, the "
              "released test set) and submission_format_5muR4s3.csv (100 rows, the platform's smoke "
              "test, whose ids are not in the released test features). Keep the 10,508-row file, or "
              "pass the right one with --submission-format.")
INSTALL_HINT = ("Install the pinned versions: uv pip install --index-strategy unsafe-best-match "
                "-r src/training/requirements-cpu.txt (README, Setup), or rerun with "
                "--skip-version-check if you mean to use another environment.")


_N_WARNINGS = 0


def _fail(msg: str) -> None:
    sys.exit(f"ERROR: {msg}")


def _warn(msg: str) -> None:
    global _N_WARNINGS
    _N_WARNINGS += 1
    print(f"WARNING: {msg}")


def _one(pattern: str) -> Path | None:
    """Return the single file in ``data/raw/`` matching ``pattern``, None if there is none.

    Stops with an error if more than one file matches.
    """
    hits = sorted(p for p in RAW.glob(pattern) if p.is_file())
    if len(hits) > 1:
        _fail(f"more than one {pattern} in {RAW}: {[h.name for h in hits]}; keep one or pass the "
              f"file explicitly")
    return hits[0] if hits else None


def _first(items, k: int = 5) -> str:
    items = list(items)
    more = f" (and {len(items) - k} more)" if len(items) > k else ""
    return ", ".join(map(str, items[:k])) + more


# --------------------------------------------------------------------------- package versions
def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pins() -> dict[str, str]:
    """Read the ``name==version`` pins of ``requirements.txt`` (so the two cannot drift apart).

    Returns:
        ``{normalised package name: pinned version}``.
    """
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].split(";", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)(\[[^\]]*\])?\s*==\s*(\S+)$", line)
        if m:
            pins[_norm(m.group(1))] = m.group(3)
    return pins


def _same_version(installed: str, pinned: str) -> bool:
    if "+" in pinned:                                  # the pin names a build, e.g. 2.11.0+cu128
        return installed == pinned
    return installed.split("+", 1)[0] == pinned        # otherwise ignore the local build tag


def check_versions() -> None:
    """Compare the installed package versions with the pins in ``requirements.txt``.

    scikit-learn and xgboost (``HARD_PINS``) must match exactly; any other difference, or a Python
    other than 3.12, is a warning.

    Returns:
        None. Prints a ``versions: OK`` line, or stops with an error listing the hard mismatches
        and missing packages.
    """
    if sys.version_info[:2] != (3, 12):
        _warn(f"Python {sys.version.split()[0]}; the model was scored with Python 3.12.")
    if not REQUIREMENTS.is_file():
        _fail(f"{REQUIREMENTS} not found, so the package versions cannot be checked. "
              f"Rerun with --skip-version-check to go ahead anyway.")
    pins = _read_pins()
    errors, warns = [], []
    for name, want in sorted(pins.items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            errors.append(f"{name} is not installed (requirements.txt pins {want})")
            continue
        if _same_version(have, want):
            continue
        if name in HARD_PINS:
            errors.append(f"{name} {have} is installed but requirements.txt pins {want}. The XGBoost and "
                          f"the distilled classifiers are pickled {name} {want} objects: with another "
                          f"version main.py fails only after the hours-long LLM pass, with an "
                          f"AttributeError (for scikit-learn < 1.8: 'LogisticRegression' object has "
                          f"no attribute 'multi_class') and no warning, because main.py silences "
                          f"warnings")
        else:
            warns.append(f"{name} {have} (requirements.txt pins {want})")
    for w in warns:
        _warn(f"version differs from requirements.txt (our verification environment): {w}")
    if errors:
        _fail("package versions:\n  - " + "\n  - ".join(errors) + "\n" + INSTALL_HINT)
    hard = " ".join(f"{n}=={pins[n]}" for n in sorted(HARD_PINS) if n in pins)
    print(f"versions: OK ({hard}; {len(pins)} pins in requirements.txt checked, "
          f"{len(warns)} warning(s))")


# --------------------------------------------------------------------------- device
_PROBE = (
    "import importlib.metadata, importlib.util, json, torch\n"
    "d = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'available': torch.cuda.is_available()}\n"
    "if d['available']:\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    d.update(n=torch.cuda.device_count(), name=p.name, mem=p.total_memory)\n"
    "d['flash_attn'] = None\n"
    "if importlib.util.find_spec('flash_attn') is not None:\n"
    "    try:\n"
    "        d['flash_attn'] = importlib.metadata.version('flash_attn')\n"
    "    except Exception:\n"
    "        d['flash_attn'] = 'unknown version'\n"
    "print(json.dumps(d))\n"
)


def _warn_flash_attn(version: str | None) -> None:
    if version is None:
        return
    _warn(f"flash_attn {version} is installed. main.py then asks transformers for "
          f"attn_implementation='flash_attention_2' first. flash-attn is neither in "
          f"requirements.txt nor in the competition runtime's lock file at commit ea9a817, and every "
          f"run in reports/gpu_noise/ used PyTorch SDPA. How much flash-attn moves the predictions "
          f"has not been measured. For the SDPA attention path, run in an environment without "
          f"flash_attn.")


def check_device(env: dict, allow_cpu: bool) -> None:
    """Report the torch build and GPU, and stop if no GPU is visible unless CPU use is allowed.

    torch is queried in a short child process, so this process never holds a CUDA context (it
    would sit on GPU memory for the whole run). Warns on a GPU under 34 GiB and if flash_attn is
    installed.

    Args:
        env: environment for the child process (the one main.py will get).
        allow_cpu: True with ``--cpu``: a machine without a GPU is accepted.

    Returns:
        None. Prints a ``device:`` line; stops with an error if torch cannot be imported or no
        GPU is visible without ``--cpu``.
    """
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
                           timeout=600)
    except subprocess.TimeoutExpired:
        _fail("the torch device probe did not finish within 10 minutes")
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        _fail("could not import torch to check the device:\n" + (r.stderr.strip()[-1500:] or "no output")
              + "\n" + INSTALL_HINT)
    d = json.loads(lines[-1])
    desc = f"torch {d['torch']}, built for CUDA {d['cuda']}, torch.cuda.is_available()={d['available']}"
    if d["available"]:
        gib = d["mem"] / 2**30
        print(f"device: {desc}; GPU 0 of {d['n']}: {d['name']}, {gib:.1f} GiB")
        if gib < 34:
            _warn(f"GPU 0 has {gib:.1f} GiB. main.py scores in batches of 8 and needed up "
                  f"to 33.4 GiB in our worst-case probe (8 x 8,192 tokens); a smaller card is "
                  f"likely to run out of memory on long transcripts. The documented minimum is "
                  f"40 GB.")
        _warn_flash_attn(d.get("flash_attn"))
        return
    if not allow_cpu:
        _fail(f"no GPU is visible ({desc}). A CPU run of the full test set would take days. Pass "
              f"--cpu to run on CPU anyway (fine for a handful of rows, e.g. data/examples/new_data).")
    print(f"device: {desc}; running on CPU (--cpu)")


def pin_scored_env(env: dict, allow_override: bool) -> None:
    """Set ``LLMKT_MAX_SEQ_LEN`` and ``LLMKT_KEEP_HEADER`` to the scored values in ``env``.

    main.py only ``setdefault()``s them, so a value left in the shell would otherwise change the
    predictions.

    Args:
        env: the environment main.py will run with; modified in place.
        allow_override: True with ``--allow-env-override``: keep a differing value and warn.

    Returns:
        None.
    """
    for key, scored in SCORED_ENV.items():
        have = env.get(key)
        if have is None or have == scored:
            env[key] = scored
        elif allow_override:
            _warn(f"{key}={have} from your environment is passed to main.py (--allow-env-override); "
                  f"the scored run used {key}={scored}, so the predictions may differ from it")
        else:
            print(f"note: {key}={have} in your environment is replaced by {scored}, the scored value, "
                  f"for main.py; pass --allow-env-override to keep yours")
            env[key] = scored


# --------------------------------------------------------------------------- base model
#: Resolves Qwen/Qwen3-8B in the local Hugging Face cache the way main.py's offline load does
#: (transformers' cached_file with local_files_only=True). Reads file paths only; loads no weights.
_CACHE_PROBE = """
import json, sys
from pathlib import Path
repo = sys.argv[1]
try:
    from huggingface_hub import constants
    from transformers.utils import cached_file
    from transformers.utils import hub as tf_hub
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.exit(0)
def get(name):
    try:
        return cached_file(repo, name, local_files_only=True)
    except Exception:
        return None
cache = Path(getattr(tf_hub, "TRANSFORMERS_CACHE", None) or constants.HF_HUB_CACHE)
repo_dir = cache / ("models--" + repo.replace("/", "--"))
snap_dir = repo_dir / "snapshots"
d = {"cache": str(cache), "has_ref": (repo_dir / "refs" / "main").is_file(),
     "snapshots": sorted(p.name for p in snap_dir.iterdir() if p.is_dir()) if snap_dir.is_dir() else [],
     "path": None, "revision": None, "missing": []}
cfg = get("config.json")
if cfg:
    folder = Path(cfg).parent
    d["path"] = str(folder)
    d["revision"] = folder.name if folder.parent.name == "snapshots" else None
    index = get("model.safetensors.index.json")
    if index:
        shards = sorted(set(json.loads(Path(index).read_text())["weight_map"].values()))
        d["missing"] = [s for s in shards if not get(s)]
    elif not get("model.safetensors"):
        d["missing"] = ["model.safetensors.index.json"]
print(json.dumps(d))
"""


def _probe_hf_cache(env: dict) -> dict:
    """Run ``_CACHE_PROBE`` offline in a child process and return its JSON result.

    Returns:
        The probe's dict (``cache``, ``has_ref``, ``snapshots``, ``path``, ``revision``,
        ``missing``), or ``{"error": ...}`` if the probe failed.
    """
    child_env = dict(env, HF_HUB_OFFLINE="1")
    try:
        r = subprocess.run([sys.executable, "-c", _CACHE_PROBE, BASE_REPO], env=child_env,
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"error": "the cache probe did not finish within 10 minutes"}
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        return {"error": r.stderr.strip()[-500:] or "no output"}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"error": "unexpected probe output"}


def describe_base_model(env: dict) -> None:
    """Report where main.py will find Qwen3-8B, and stop if it is nowhere.

    Checks main.py's locations in main.py's order: the platform mount, the relative
    ``./huggingface_models`` mount (never present in the temporary directory),
    ``LLMKT_BASE_DIR``, then the local Hugging Face cache, read offline.

    Args:
        env: the environment main.py will run with.

    Returns:
        None. Prints a ``base model:`` line; warns about a cached snapshot at another revision;
        stops with an error (and a fix) if the model is in none of the places or a weight file is
        missing.
    """
    if (PLATFORM_BASE / "config.json").exists():
        print(f"base model: {PLATFORM_BASE} (platform mount)")
        return
    base = env.get("LLMKT_BASE_DIR", "")
    if base and (Path(base) / "config.json").exists():
        print(f"base model: {base} (LLMKT_BASE_DIR)")
        return
    if base:
        _warn(f"LLMKT_BASE_DIR={base} holds no config.json, so main.py ignores it and tries the "
              f"local Hugging Face cache"
              + (f"; {LOCAL_BASE} holds the model but is not used while LLMKT_BASE_DIR is set: "
                 f"unset it" if (LOCAL_BASE / "config.json").exists() else ""))
    d = _probe_hf_cache(env)
    if "error" in d:
        _warn(f"base model Qwen/Qwen3-8B is not at {LOCAL_BASE} or LLMKT_BASE_DIR, and the local "
              f"Hugging Face cache could not be checked ({d['error']}). main.py stops with an "
              f"OSError if the model is not in the cache either. Download it as in README, Setup.")
        return
    where = (f"is not at {PLATFORM_BASE}, {LOCAL_BASE} or LLMKT_BASE_DIR ({base or 'not set'})")
    if d["path"] is None:
        hint = ""
        if d["snapshots"] and not d["has_ref"]:
            ref = Path(d["cache"]) / f"models--{BASE_REPO.replace('/', '--')}" / "refs" / "main"
            rev = BASE_REVISION if BASE_REVISION in d["snapshots"] else d["snapshots"][0]
            hint = (f" The cache holds snapshot(s) {_first(d['snapshots'])} but no {ref}, so an "
                    f"offline load cannot resolve the model (a download pinned to a commit hash "
                    f"may not write it). Write the snapshot's commit hash into that file "
                    f"(mkdir -p {ref.parent} && printf %s {rev} > {ref}).")
        _fail(f"base model Qwen/Qwen3-8B {where}, nor in the Hugging Face cache {d['cache']}. "
              f"main.py never downloads, so it would stop with an OSError.{hint} "
              f"{'Or put' if hint else 'Put'} the model at {LOCAL_BASE} or set LLMKT_BASE_DIR "
              f"(README, Setup).")
    if d["missing"]:
        _fail(f"base model Qwen/Qwen3-8B {where}, and its copy in the Hugging Face cache, "
              f"{d['path']}, lacks {_first(d['missing'])}, so main.py's offline load would fail. "
              f"Download it again as in README, Setup.")
    print(f"base model: {d['path']} (Hugging Face cache, offline)")
    if d["revision"] != BASE_REVISION:
        _warn(f"the cached base model is revision {d['revision'] or 'unknown'}, not "
              f"{BASE_REVISION}, the base the adapter was trained and scored on. Other revisions "
              f"load, but the predictions move (README, Setup).")


def verify_assets(sub: Path) -> None:
    """Check every file listed in ``assets/manifest_sha256.txt`` against its sha256.

    Args:
        sub: the submission folder (``main.py`` + ``assets/``).

    Returns:
        None. Prints an ``assets: OK`` line with main.py's sha256; stops with an error if a file
        is missing or differs, and only warns if the manifest itself is absent.
    """
    manifest = sub / "assets" / "manifest_sha256.txt"
    if not manifest.is_file():
        _warn(f"{manifest} not found; asset checksums not verified")
        return
    n, bad = 0, []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.strip().split(None, 1)
        path = sub / "assets" / rel
        if not path.is_file():
            bad.append(f"{rel} (missing)")
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            bad.append(f"{rel} (sha256 differs)")
        n += 1
    if bad:
        _fail(f"asset manifest: {len(bad)} file(s) do not match {manifest}: {_first(bad, 10)}")
    main_sha = hashlib.sha256((sub / "main.py").read_bytes()).hexdigest()
    print(f"assets: OK ({n} files match assets/manifest_sha256.txt; main.py sha256 {main_sha})")


# --------------------------------------------------------------------------- inputs
def read_features(path: Path) -> pd.DataFrame:
    """Read and check the features CSV.

    Every required column must be present; ``response_id`` must be non-empty and unique; the
    other fields must be non-empty as main.py parses them (pandas' missing-value words count as
    empty).

    Args:
        path: CSV with response_id, session_id, learning_objective_id, learning_objective.

    Returns:
        The four required columns as strings, in file order. Stops with an error on any problem.
    """
    feats = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in feats.columns]
    if missing:
        _fail(f"{path} is missing columns {missing}; it needs {REQUIRED}")
    if feats.empty:
        _fail(f"{path} has no rows")
    feats = feats[REQUIRED].copy()
    empty = feats.index[feats["response_id"].str.strip() == ""]
    if len(empty):
        _fail(f"{path}: empty response_id in {len(empty)} row(s) (1-based, header excluded): "
              f"{_first(empty + 1, 10)}")
    dup = feats["response_id"][feats["response_id"].duplicated()]
    if len(dup):
        _fail(f"{path}: response_id must be unique; {dup.nunique()} id(s) repeat: "
              f"{_first(dup.unique(), 10)}")
    # The other fields, read back exactly as main.py reads test_features.csv.
    buf = io.StringIO()
    feats.to_csv(buf, index=False)
    seen = pd.read_csv(io.StringIO(buf.getvalue()), dtype=str)
    for col in REQUIRED[1:]:
        bad = seen.index[seen[col].isna() | (seen[col].fillna("").str.strip() == "")]
        if len(bad):
            _fail(f"{path}: {col} is empty, or a word pandas reads as missing (NA, None, null, nan, "
                  f"n/a, ...), in {len(bad)} row(s) (1-based, header excluded): {_first(bad + 1, 10)}. "
                  f"The unchanged main.py cannot score such rows.")
    return feats


def _ids_survive(ids: list[str]) -> bool:
    """Tell whether main.py would read these response ids back unchanged.

    Args:
        ids: the response ids, in order.

    Returns:
        True when both of main.py's reads give back the same strings: ``submission_format.csv``
        (no dtype) and ``test_features.csv`` (``dtype=str``).
    """
    buf = io.StringIO()
    pd.DataFrame({"response_id": ids, "probability": 0.5}).to_csv(buf, index=False)
    text = buf.getvalue()
    for kw in ({}, {"dtype": str}):
        back = pd.read_csv(io.StringIO(text), **kw)["response_id"].tolist()
        if len(back) != len(ids) or any(not isinstance(b, str) or b != a for a, b in zip(ids, back)):
            return False
    return True


def _format_ids(path: Path) -> pd.DataFrame:
    fmt = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "response_id" not in fmt.columns:
        _fail(f"{path} has no response_id column")
    return fmt


def choose_format(explicit: Path | None, use_raw: bool, feat_ids: set) -> tuple[Path | None, pd.DataFrame | None]:
    """Pick the format file that sets which rows are written, and in which order.

    A format file qualifies when it has rows, no repeated ids and only ids from the features.
    Without ``--submission-format``, the qualifying ``data/raw/submission_format*.csv`` with the
    most rows wins.

    Args:
        explicit: the ``--submission-format`` path, or None.
        use_raw: look in ``data/raw/`` when ``explicit`` is None (True when ``--features`` was not
            given).
        feat_ids: the response ids of the features file.

    Returns:
        ``(path, frame)`` of the chosen file, or ``(None, None)`` to keep the features' order.
        Stops with an error if the given file, or every candidate in ``data/raw/``, fails the check.
    """
    def problem(path: Path, fmt: pd.DataFrame) -> str | None:
        ids = fmt["response_id"]
        if fmt.empty:
            return "has no rows"
        if ids.duplicated().any():
            return f"repeats {ids[ids.duplicated()].nunique()} response_id(s)"
        outside = (~ids.isin(feat_ids)).sum()
        if outside:
            return f"has {outside} of {len(ids)} response_ids that are not in the features"
        return None

    if explicit is not None:
        fmt = _format_ids(explicit)
        why = problem(explicit, fmt)
        if why:
            note = f" {SMOKE_NOTE}" if explicit.name.startswith("submission_format") else ""
            _fail(f"--submission-format {explicit} {why}.{note}")
        print(f"format: {explicit} ({len(fmt)} rows)")
        return explicit, fmt
    cands = sorted(p for p in RAW.glob("submission_format*.csv") if p.is_file()) if use_raw else []
    if not cands:
        print("format: none given; rows are written in the order of the features file")
        return None, None
    good, rejected = [], []
    for p in cands:
        fmt = _format_ids(p)
        why = problem(p, fmt)
        (rejected.append(f"{p.name} ({len(fmt)} rows) {why}") if why else good.append((len(fmt), p.stat().st_size, p, fmt)))
    if not good:
        _fail("no usable submission_format*.csv in data/raw:\n  - " + "\n  - ".join(rejected)
              + f"\n{SMOKE_NOTE}")
    good.sort(key=lambda t: (t[0], t[1]), reverse=True)
    n, _, path, fmt = good[0]
    others = [f"{p.name} ({k} rows)" for k, _, p, _ in good[1:]] + rejected
    print(f"format: {path.name} ({n} rows)" + (f"; not used: {'; '.join(others)}" if others else ""))
    return path, fmt


def _read_member(z: zipfile.ZipFile, m: zipfile.ZipInfo) -> bytes:
    """Read one zip member, skipping the CRC check if the stored CRC is wrong."""
    try:
        return z.open(m).read()
    except zipfile.BadZipFile:                         # the released test zip has one stale CRC record
        f = z.open(m)
        f._expected_crc = None
        return f.read()


def iter_transcripts(src: Path, sessions: list[str]):
    """Yield the raw transcript of each requested session that has a file.

    Args:
        src: a folder of ``<session_id>.csv`` files, or a zip containing them (at any depth).
        sessions: the session ids to look for.

    Yields:
        ``(session_id, file bytes)``, in the order of ``sessions``.
    """
    want = set(sessions)
    if src.is_dir():
        for s in sessions:
            p = src / f"{s}.csv"
            if p.is_file():
                yield s, p.read_bytes()
        return
    if not (src.is_file() and zipfile.is_zipfile(src)):
        _fail(f"--transcripts {src} is neither a folder nor a zip file")
    found = {}
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            name = Path(m.filename).name
            if name.endswith(".csv") and name[:-4] in want:
                found[name[:-4]] = m                    # if a name repeats, the later member wins
        for s in sessions:
            if s in found:
                yield s, _read_member(z, found[s])


def stage_transcripts(src: Path, sessions: list[str], dest: Path | None,
                      incomplete_hint: str | None = None) -> None:
    """Check every requested transcript and copy it, with normalised roles, into ``dest``.

    Stops before any model work if a transcript cannot be read or would crash main.py (no tutor
    line), if no requested session has a transcript at all, or, when ``incomplete_hint`` is given,
    if any session is absent. Format problems (utterance ids, timestamps, unknown roles) are
    warnings.

    Args:
        src: folder or zip of transcripts.
        sessions: the requested session ids.
        dest: folder for the copies, or None to check only (``--check``).
        incomplete_hint: path of a ``test_transcripts*.zip`` that sits next to the default folder;
            set only when that folder was picked, so a partly unpacked folder is caught.

    Returns:
        None. Writes ``<dest>/<session_id>.csv`` for each usable transcript (the original bytes
        unless the roles were changed).
    """
    present, zero, renorm, unknown, notutor, broken = set(), [], 0, set(), [], []
    badid, dupid, badts, na_words, na_tx = [], [], [], 0, 0
    for sid, payload in iter_transcripts(src, sessions):
        if not payload.strip():
            zero.append(sid)                           # 0-byte: treated as missing
            continue
        try:
            tx = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            broken.append(f"{sid} ({type(e).__name__})")
            continue
        miss = [c for c in TRANSCRIPT_REQUIRED if c not in tx.columns]
        if miss:
            broken.append(f"{sid} (missing columns {miss})")
            continue
        present.add(sid)
        if len(tx):
            # Format checks: warnings only. main.py reads these cells with pandas' defaults.
            uid = pd.to_numeric(tx["utterance_id"], errors="coerce")
            if uid.isna().any() or (uid.dropna() % 1 != 0).any():
                badid.append(sid)
            if uid.dropna().duplicated().any():
                dupid.append(sid)
            if not tx["timestamp"].str.strip().str.fullmatch(TIMESTAMP_RE).all():
                badts.append(sid)
            seen = pd.read_csv(io.BytesIO(payload), dtype=str)    # as main.py reads it
            n_na = int((seen["content"].isna() & tx["content"].str.strip().ne("")).sum())
            if n_na:
                na_words += n_na
                na_tx += 1
        role = tx["role"].str.strip().str.lower()
        unknown.update(set(role) - KNOWN_ROLES - {""})
        if len(tx) and not role.eq("tutor").any():
            notutor.append(sid)
        changed = not role.equals(tx["role"])
        renorm += changed
        if dest is not None:
            if changed:
                tx["role"] = role
                tx.to_csv(dest / f"{sid}.csv", index=False)
            else:
                (dest / f"{sid}.csv").write_bytes(payload)   # untouched bytes, as the platform saw them
    if broken:
        _fail(f"{len(broken)} transcript(s) cannot be read as CSV with columns "
              f"{TRANSCRIPT_REQUIRED}: {_first(broken, 10)}")
    if notutor:
        _fail(f"{len(notutor)} transcript(s) have utterances but no 'tutor' line, which makes "
              f"main.py's feature code fail after the LLM pass: {_first(notutor, 20)}. Remove those "
              f"rows from the features file, or fix the role column (tutor/student/background).")
    if zero:
        _warn(f"{len(zero)} transcript file(s) are empty (0 bytes or only whitespace) and are "
              f"treated as missing: "
              f"{_first(zero)}")
    if renorm:
        print(f"note: role was stripped/lower-cased in {renorm} transcript(s) (in the temporary copy)")
    if unknown:
        _warn(f"roles other than tutor/student/background are ignored by the model: "
              f"{_first(sorted(unknown))}")
    if badid:
        _warn(f"{len(badid)} transcript(s) have an utterance_id that is not a whole number: "
              f"{_first(badid)}. main.py orders utterances by utterance_id: its XGBoost half moves a value "
              f"that is not a number (an empty cell included) to the end, and a text value makes its "
              f"LLM half sort the whole column as text (10 before 2).")
    if dupid:
        _warn(f"{len(dupid)} transcript(s) repeat an utterance_id: {_first(dupid)}. main.py orders "
              f"utterances by utterance_id, and the order of repeated ids is not defined.")
    if badts:
        _warn(f"{len(badts)} transcript(s) have a timestamp that is not HH:MM:SS (elapsed time): "
              f"{_first(badts)}. The XGBoost half's timing features read such a value as missing unless "
              f"it is three colon-separated whole numbers.")
    if na_words:
        print(f"note: {na_words} utterance(s) in {na_tx} transcript(s) consist only of a word pandas "
              f"reads as missing (NA, None, null, ...); main.py reads them as empty, as in the "
              f"scored run, so the LLM prompt leaves them out")
    absent = [s for s in sessions if s not in present]
    if sessions and not present:
        nested = sorted(p.name for p in src.iterdir() if p.is_dir()) if src.is_dir() else []
        _fail(f"none of the {len(sessions)} requested session(s) has a usable transcript in {src}"
              + (f" (it contains sub-folder(s) {_first(nested)}; the <session_id>.csv files must sit "
                 f"directly in {src})" if nested else "")
              + ". A folder must hold <session_id>.csv files at its top level, and a zip must contain "
                "them as members. A row that really has no transcript can be given a header-only "
                "file (session_id,utterance_id,role,content,timestamp).")
    if absent and incomplete_hint:
        _fail(f"{len(absent)} of {len(sessions)} session(s) have no usable transcript in {src}, and "
              f"{incomplete_hint} is also in data/raw, so the folder looks partly unpacked. Delete the "
              f"folder (the zip is then used) or pass --transcripts {incomplete_hint}. First few: "
              f"{absent[:5]}")
    if absent:
        _warn(f"{len(absent)} session(s) have no usable transcript; for those rows the LLM "
              f"half falls back to its Platt map at 0.5 (about 0.61); the XGBoost half still scores the "
              f"objective. First few: {absent[:5]}")
    print(f"transcripts: {len(present)} of {len(sessions)} session(s) found")


# --------------------------------------------------------------------------- output
def check_out(out: Path, create: bool) -> None:
    """Stop in seconds, not after the LLM pass, if ``--out`` cannot be written.

    Args:
        out: the output file path.
        create: create the output folder now; with False (``--check``) nothing is created and the
            nearest existing ancestor folder is checked instead.

    Returns:
        None. Stops with an error if ``out`` is a folder or not writable.
    """
    if out.is_dir():
        _fail(f"--out {out} is a folder; pass a file path such as {out / 'submission.csv'}")
    if create:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            _fail(f"--out: cannot create the folder {out.parent}: {e}")
    anc = out.parent
    while not anc.exists() and anc != anc.parent:
        anc = anc.parent
    if not anc.is_dir():
        _fail(f"--out {out}: {anc} is not a folder")
    if not os.access(anc, os.W_OK | os.X_OK) or (out.exists() and not os.access(out, os.W_OK)):
        _fail(f"--out {out} is not writable")


def write_out(src: Path, out: Path) -> None:
    """Copy the finished predictions to ``--out``, with a fallback copy if that fails.

    The fallback (``submission_rescued_<pid>.csv`` in ``data/processed/``, the working directory
    or the system temporary folder) is written before the temporary directory is removed, so an
    I/O error never discards the run.

    Args:
        src: the predictions in the temporary directory.
        out: the requested output file.

    Returns:
        None. Stops with an error (naming the fallback file, if any) when ``out`` cannot be written.
    """
    try:
        shutil.copyfile(src, out)                       # copyfile, not copy2: a folder is an error
        return
    except OSError as e:
        err = e
    name = f"submission_rescued_{os.getpid()}.csv"
    for folder in (REPO / "data" / "processed", Path.cwd(), Path(tempfile.gettempdir())):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, folder / name)
        except OSError:
            continue
        _fail(f"could not write {out}: {err}. The predictions were saved to {folder / name}")
    _fail(f"could not write {out}: {err}, nor a fallback copy; the predictions are lost")


# --------------------------------------------------------------------------- main
def main() -> None:
    """Check the inputs and environment, stage them for main.py, run it and write the predictions.

    Returns:
        None. Writes ``--out`` (default ``data/processed/submission.csv``) with columns
        ``response_id, probability`` in the format file's row order. With ``--check`` only the
        checks run and nothing is written. Any failed check exits with ``ERROR: ...``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--transcripts", type=Path, default=None)
    ap.add_argument("--submission-format", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "processed" / "submission.csv")
    ap.add_argument("--submission-dir", type=Path, default=REPO / "models" / "submission")
    ap.add_argument("--check", action="store_true", help="preflight only; loads no model")
    ap.add_argument("--cpu", action="store_true", help="allow running without a GPU")
    ap.add_argument("--skip-version-check", action="store_true")
    ap.add_argument("--allow-env-override", action="store_true",
                    help="pass your own LLMKT_MAX_SEQ_LEN / LLMKT_KEEP_HEADER to main.py")
    args = ap.parse_args()
    check_out(args.out, create=not args.check)

    sub = args.submission_dir.resolve()
    if not (sub / "main.py").is_file() or not (sub / "assets").is_dir():
        _fail(f"{sub} does not contain main.py and assets/ (--submission-dir)")

    features = args.features or _one("test_features*.csv")
    incomplete_hint = None
    if args.transcripts is not None:
        transcripts = args.transcripts
    else:
        tdir_default = RAW / "test_transcripts"
        zips = sorted(p for p in RAW.glob("test_transcripts*.zip") if p.is_file())
        if tdir_default.is_dir() and any(p.is_file() for p in tdir_default.glob("*.csv")):
            transcripts = tdir_default
            incomplete_hint = str(zips[0]) if zips else None
        else:
            transcripts = _one("test_transcripts*.zip")
            if tdir_default.is_dir() and transcripts is not None:
                print(f"note: {tdir_default} holds no *.csv at its top level; using {transcripts}")
    if features is None or transcripts is None:
        _fail("no input: put the competition's test files in data/raw/ or pass --features and --transcripts")
    if not Path(features).is_file():
        _fail(f"--features {features} not found")
    if not Path(transcripts).exists():
        _fail(f"--transcripts {transcripts} not found")
    print(f"transcripts: reading {transcripts}")

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"                # keep __pycache__ out of models/submission/assets
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # measured bit-identical
    if "LLMKT_BASE_DIR" not in env and (LOCAL_BASE / "config.json").exists():
        env["LLMKT_BASE_DIR"] = str(LOCAL_BASE)
    # main.py runs in a temporary directory, so resolve relative paths against the caller's cwd
    for k in ("LLMKT_BASE_DIR", "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        if env.get(k) and not os.path.isabs(env[k]):
            env[k] = os.path.abspath(env[k])
    pin_scored_env(env, args.allow_env_override)

    # 1. inputs (cheap, so a bad file stops the run in seconds)
    feats = read_features(features)
    fmt_path, fmt = choose_format(args.submission_format, args.features is None, set(feats["response_id"]))
    out_ids = list(fmt["response_id"]) if fmt is not None else list(feats["response_id"])
    if fmt is not None and len(out_ids) < len(feats):
        print(f"note: the format lists {len(out_ids)} of the {len(feats)} feature rows; only those "
              f"are written")
    remap = not (_ids_survive(list(feats["response_id"])) and _ids_survive(out_ids))
    if remap:
        print("note: some response_ids would not survive main.py's CSV parsing (e.g. 1 or 007); "
              "they are replaced by rid_<n> for the run and restored in the output")
    print(f"features: {len(feats)} row(s), {feats['session_id'].nunique()} session(s)")
    # 2. environment
    if args.skip_version_check:
        print("versions: not checked (--skip-version-check)")
    else:
        check_versions()
    check_device(env, args.cpu)
    describe_base_model(env)
    sessions = list(dict.fromkeys(feats["session_id"]))

    if args.check:
        verify_assets(sub)
        stage_transcripts(Path(transcripts), sessions, None, incomplete_hint)
        if _N_WARNINGS:
            print(f"CHECK PASSED WITH {_N_WARNINGS} WARNING{'S' if _N_WARNINGS != 1 else ''} "
                  f"(no model was loaded)")
        else:
            print("CHECK PASSED (no model was loaded)")
        return

    to_tmp = {rid: f"rid_{i}" for i, rid in enumerate(feats["response_id"])} if remap else None
    with tempfile.TemporaryDirectory(prefix="tta_inference_") as tmp:
        work = Path(tmp)
        (work / "main.py").symlink_to(sub / "main.py")  # main.py finds assets/ via its real path
        data = work / "data"
        data.mkdir()
        tdir = data / "test_transcripts"
        tdir.mkdir()
        stage_transcripts(Path(transcripts), sessions, tdir, incomplete_hint)
        staged = feats.copy()
        if remap:
            staged["response_id"] = staged["response_id"].map(to_tmp)
        staged.to_csv(data / "test_features.csv", index=False)
        if fmt is not None and not remap:
            shutil.copy2(fmt_path, data / "submission_format.csv")
        elif fmt is not None:
            f2 = fmt.copy()
            f2["response_id"] = f2["response_id"].map(to_tmp)
            f2.to_csv(data / "submission_format.csv", index=False)
        else:
            pd.DataFrame({"response_id": staged["response_id"], "probability": 0.5}).to_csv(
                data / "submission_format.csv", index=False)

        sys.stdout.flush()                              # keep our lines ahead of main.py's in a log
        rc = subprocess.call([sys.executable, "main.py"], cwd=work, env=env)
        if rc != 0:
            _fail(f"main.py failed with exit code {rc}")
        result = work / "submission.csv"
        if not result.is_file():
            _fail("main.py exited without writing submission.csv")
        if remap:
            res = pd.read_csv(result, dtype=str, keep_default_na=False)
            back = {v: k for k, v in to_tmp.items()}
            res["response_id"] = res["response_id"].map(back)
            if res["response_id"].isna().any():
                _fail("main.py wrote response_ids that are not in the input")
            result = work / "submission_restored.csv"
            res.to_csv(result, index=False)
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)   # checked at start; may have gone since
        except OSError:
            pass                                        # write_out reports it and saves a fallback copy
        write_out(result, args.out)

    written = pd.read_csv(args.out, dtype=str, keep_default_na=False)
    if list(written["response_id"]) != out_ids:
        _fail(f"{args.out} does not list the expected response_ids in the expected order")
    p = pd.to_numeric(written["probability"], errors="coerce")
    if p.isna().any() or not p.between(0, 1).all():
        _fail(f"{args.out} has missing or out-of-range probabilities")
    if (p == 0.5).all():
        _warn("every probability is exactly 0.5, which means no row was matched to a "
              "prediction. Check the response_ids of the features and the format file.")
    try:
        shown = args.out.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        shown = args.out
    print(f"wrote {shown} ({len(written)} rows)"
          + (f"; the wrapper printed {_N_WARNINGS} warning(s) above" if _N_WARNINGS else ""))


if __name__ == "__main__":
    main()
