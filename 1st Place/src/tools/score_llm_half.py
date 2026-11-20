#!/usr/bin/env python
"""Score the LLM half of the shipped model on one device and record the hardware it ran on.

Used to measure how much the predictions move from one GPU to another (reports/gpu_noise/report.md).
The XGBoost half is computed on the CPU and is identical on every machine, so the LLM half is the
only part that can differ, and this script scores just that half.

It calls ``main.run_llmkt`` from the shipped ``main.py``: the same model loader, CPU merge, prompts,
8,192-token header-preserving truncation, batching and calibrator. Nothing in ``main.py`` or
``assets/`` is changed. The only addition is a thin wrapper around ``llmkt_infer.score_prompts``
that keeps a copy of the raw probabilities (before the Platt map). ``--batch-size`` overrides the
shipped batch of 8 for cards too small to hold it; a different batch size can move some rows by
one bf16 step (README, Run inference; reports/gpu_noise/report.md), so the run records which batch
size it used.

    python src/tools/score_llm_half.py --features data/raw/test_features*.csv \
        --transcripts data/raw/test_transcripts*.zip --tag a5000 --out-dir reports/gpu_noise

Writes ``<out-dir>/<tag>.csv`` (response_id, p_raw, z_raw, p_llm) and ``<out-dir>/<tag>.json``
(device, driver, CUDA, library versions, batch size, wall time). ``z_raw`` is
logit_True - logit_False, the quantity that moves by bf16 steps.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SUB = REPO / "models" / "submission"                # the shipped submission tree (the scored one with re-saved vocabularies and edited comments; models/README.md)
sys.dont_write_bytecode = True          # importing main.py must not leave __pycache__ in the folder/assets
REQUIRED = ["response_id", "session_id", "learning_objective_id", "learning_objective"]


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _cudnn_version(torch):
    """Return the cuDNN version, or the error text where cuDNN cannot initialise.

    cuDNN 9.x does not initialise on GPUs below compute capability 7.5 (e.g. V100), and torch
    raises when asked for the version there.
    """
    try:
        return torch.backends.cudnn.version()
    except RuntimeError as e:
        return f"unavailable: {e}"


def hardware(device: str) -> dict:
    """Describe the machine and software stack a run uses.

    Args:
        device: "cuda" or "cpu".

    Returns:
        A dict of host, CPU, RAM, OS, Python, torch/CUDA/cuDNN, transformers and peft versions,
        and, on CUDA, the GPU name, memory, compute capability and driver version.
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    import peft  # noqa: PLC0415

    info = {
        "host": socket.gethostname(),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
        "cpu_threads": os.cpu_count(),
        "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": _cudnn_version(torch),
        "transformers": transformers.__version__, "peft": peft.__version__,
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
        "visible_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device": device,
    }
    if device.startswith("cuda"):
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        info.update({
            "gpu": p.name, "gpu_mem_gb": round(p.total_memory / 2**30, 1),
            "compute_capability": f"{p.major}.{p.minor}", "sm_count": p.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "driver": (_sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
                       .splitlines() or [""])[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        })
    else:
        info.update({"gpu": "none (CPU)", "gpu_mem_gb": 0, "compute_capability": "", "driver": ""})
    return info


def stage(features: Path, transcripts: Path, rows: int, work: Path) -> pd.DataFrame:
    """Lay out the ``data/`` folder that ``main.py`` reads, in a scratch directory.

    Args:
        features: test features CSV (needs response_id, session_id, learning_objective_id,
            learning_objective).
        transcripts: folder of ``<session_id>.csv`` files, or the released zip.
        rows: if non-zero, keep a fixed sample of whole sessions with about this many rows
            (the same sample on every machine); 0 keeps all rows.
        work: empty scratch directory; ``work/data/`` is created in it.

    Returns:
        The selected feature rows. Writes ``work/data/test_features.csv``,
        ``work/data/submission_format.csv`` (all 0.5) and ``work/data/test_transcripts/``
        (a link to the folder, or the needed members extracted from the zip). Exits if a
        required column is missing.
    """
    feats = pd.read_csv(features, dtype=str)
    missing = [c for c in REQUIRED if c not in feats.columns]
    if missing:
        sys.exit(f"--features is missing columns {missing}")
    if rows:
        # a fixed sample of whole sessions, the same on every machine
        sess = sorted(feats.session_id.unique())
        rng = np.random.default_rng(20261017)
        keep = set(rng.permutation(sess)[: max(1, int(round(rows * len(sess) / len(feats))))])
        feats = feats[feats.session_id.isin(keep)]
    data = work / "data"
    data.mkdir(parents=True)
    feats[REQUIRED].to_csv(data / "test_features.csv", index=False)
    pd.DataFrame({"response_id": feats.response_id, "probability": 0.5}).to_csv(
        data / "submission_format.csv", index=False)
    if transcripts.is_dir():
        (data / "test_transcripts").symlink_to(transcripts.resolve(), target_is_directory=True)
    else:
        # extract only the sessions we need; the released zip's one stale CRC record
        # (hkohkub.csv) is harmless, so read members individually
        out = data / "test_transcripts"
        out.mkdir()
        want = {f"{s}.csv" for s in feats.session_id.unique()}
        with zipfile.ZipFile(transcripts) as z:
            for m in z.infolist():
                name = Path(m.filename).name
                if name in want:
                    try:
                        payload = z.open(m).read()
                    except zipfile.BadZipFile:           # stale CRC record: payload is intact
                        f = z.open(m)
                        f._expected_crc = None
                        payload = f.read()
                    (out / name).write_bytes(payload)
    return feats


def main() -> None:
    """Score the LLM half of the shipped model on this device through ``main.run_llmkt``.

    Returns:
        Nothing; writes ``<out-dir>/<tag>.csv`` (response_id, p_raw, z_raw, p_llm) and
        ``<out-dir>/<tag>.json`` (hardware, versions, batch size, timings) and prints the JSON.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True, help="directory of <session_id>.csv, or the zip")
    ap.add_argument("--tag", required=True, help="name of this run, e.g. a100-80gb")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "gpu_noise")
    ap.add_argument("--rows", type=int, default=0, help="score a fixed sample of about N rows (0 = all)")
    ap.add_argument("--batch-size", type=int, default=0, help="override the shipped batch of 8 (0 = keep 8)")
    args = ap.parse_args()

    # number GPUs as nvidia-smi does (CUDA's default order is fastest-first, so on a machine with
    # mixed cards CUDA_VISIBLE_DEVICES=1 can select a different card than `nvidia-smi -i 1`)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "LLMKT_BASE_DIR" not in os.environ and (REPO / "models/huggingface_models/Qwen/Qwen3-8B/config.json").exists():
        os.environ["LLMKT_BASE_DIR"] = str(REPO / "models/huggingface_models/Qwen/Qwen3-8B")

    # main.run_llmkt sets these with setdefault() right before importing llmkt_infer, which reads
    # them at import time. We import llmkt_infer first (to wrap it), so apply main.py's own values.
    src = (SUB / "main.py").read_text()
    for k, v in re.findall(r'os\.environ\.setdefault\("(LLMKT_[A-Z_]+)", "([^"]+)"\)', src):
        os.environ.setdefault(k, v)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tta_gpu_noise_") as tmp:
        work = Path(tmp)
        feats = stage(args.features, args.transcripts, args.rows, work)
        os.chdir(work)                                   # main.py reads ./data and writes ./submission.csv

        spec = importlib.util.spec_from_file_location("tta_main", SUB / "main.py")
        tta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tta)                     # puts assets/ on sys.path
        from magnificat import llmkt_infer as li         # noqa: PLC0415
        assert li.MAX_SEQ_LEN == int(os.environ["LLMKT_MAX_SEQ_LEN"])

        captured: dict = {}
        shipped = li.score_prompts

        def score_prompts(model, tokenizer, prompts, true_token, false_token, batch_size=8, device="cuda"):
            bs = args.batch_size or batch_size
            captured["batch_size"] = bs
            captured["attn"] = getattr(model.config, "_attn_implementation", "")
            captured["n_prompts"] = len(prompts)
            t0 = time.time()
            out = shipped(model, tokenizer, prompts, true_token, false_token, batch_size=bs, device=device)
            captured["score_s"] = time.time() - t0
            captured["raw"] = list(out)
            return out

        li.score_prompts = score_prompts
        import torch  # noqa: PLC0415
        device = "cuda" if torch.cuda.is_available() else "cpu"
        meta = hardware(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        proba = tta.run_llmkt(return_proba=True)
        meta["wall_s"] = round(time.time() - t0, 1)
        meta["score_s"] = round(captured.get("score_s", float("nan")), 1)
        meta["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if device == "cuda" else 0
        meta.update({k: captured.get(k) for k in ("batch_size", "attn", "n_prompts")})
        meta.update({"tag": args.tag, "rows": len(feats), "max_seq_len": li.MAX_SEQ_LEN,
                     "keep_header": os.environ.get("LLMKT_KEEP_HEADER"),
                     "date": time.strftime("%Y-%m-%d %H:%M:%S")})

    # raw probabilities come back in prompt order; prompts are built in feature-row order and skip
    # rows without a usable transcript (those get the calibrated base rate, as in main.py)
    p_cal = proba.reindex(feats.response_id.to_numpy()).to_numpy()
    cal = tta.joblib.load(tta.ASSETS / "llmkt_calibrator.joblib")
    # recover each scored row's raw probability through the frozen map's inverse, then check it
    # against the captured raw list (they must agree; the captured list is authoritative)
    z_cal = np.log(p_cal / (1 - p_cal))
    raw_inv = 1 / (1 + np.exp(-(z_cal - cal.b) / cal.a))
    captured_raw = np.asarray(captured.get("raw", []), dtype=float)
    if len(captured_raw) == len(feats):
        assert np.allclose(raw_inv, captured_raw, atol=1e-9), "inverse-map check failed"
        raw = captured_raw
    else:                                                # some rows had no transcript
        raw = raw_inv
    out = pd.DataFrame({"response_id": feats.response_id.to_numpy(), "p_raw": raw,
                        "z_raw": np.log(raw / (1 - raw)), "p_llm": p_cal})
    out.to_csv(args.out_dir / f"{args.tag}.csv", index=False, float_format="%.10f")
    (args.out_dir / f"{args.tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir / (args.tag + '.csv')} ({len(out)} rows)")


if __name__ == "__main__":
    main()
