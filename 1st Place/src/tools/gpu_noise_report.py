#!/usr/bin/env python
"""Summarise how much the shipped model's predictions and score move from one GPU to another.

Reads every ``<tag>.csv`` / ``<tag>.json`` pair written by ``src/tools/score_llm_half.py`` (by default
from ``reports/gpu_noise/``, where ``src/training/launchers/gpu_noise/`` writes them), adds the
XGBoost half (computed on the CPU and identical on every machine, from
``models/predictions/test/shipped_components.csv``) and the released test labels, and writes a
markdown report: one row per device with its hardware, the LLM-alone and ensemble scores, and how
far its predictions sit from the reference run (our 2026-08-28 H100 run of the scored submission,
with its original asset files, whose LLM half is the ``p_llm`` column of
``shipped_components.csv``).

    python src/tools/gpu_noise_report.py --labels /path/to/test_labels.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "models" / "submission" / "assets"
W = float((ASSETS / "blend_weight.txt").read_text().strip())   # XGBoost weight, 0.43
EPS = 1e-6
ARCH = {"7.0": "Volta", "7.5": "Turing", "8.0": "Ampere", "8.6": "Ampere", "8.9": "Ada Lovelace",
        "9.0": "Hopper", "10.0": "Blackwell", "12.0": "Blackwell"}
REFERENCE = {"tag": "reference_h100pcie", "gpu": "NVIDIA H100 PCIe", "gpu_mem_gb": 79.6,
             "compute_capability": "9.0", "batch_size": 8, "torch": "2.11.0+cu128",
             "host": "n-62-11-44 (DTU HPC, LSF job 29267305)", "date": "2026-08-28",
             "note": "the shipped main.py plus an 11-line side-output block that writes the two halves; LLM half = shipped_components.csv p_llm"}


def score(y, p):
    """Log loss and AUROC x 100 of probabilities ``p`` (clipped to [1e-6, 1 - 1e-6]) for labels ``y``."""
    p = np.clip(p, EPS, 1 - EPS)
    return log_loss(y, p, labels=[0, 1]), 100 * roc_auc_score(y, p)


def main() -> None:
    """Score every device run against the labels and the reference run, and write the report.

    Returns:
        Nothing; prints the report and writes it to ``--out`` (default
        ``<runs>/report.md``), with the per-run numbers in a CSV of the same name.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=REPO / "reports" / "gpu_noise")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--components", type=Path,
                    default=REPO / "models" / "predictions" / "test" / "shipped_components.csv")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--exclude", nargs="*", default=["test_"], help="tag prefixes to skip")
    args = ap.parse_args()

    comp = pd.read_csv(args.components)
    lab = pd.read_csv(args.labels)
    sys.dont_write_bytecode = True               # importing magnificat must not leave __pycache__ in assets/
    sys.path.insert(0, str(ASSETS))              # the calibrator pickle needs magnificat
    cal = joblib.load(ASSETS / "llmkt_calibrator.joblib")
    cal_a, cal_b = cal.a, cal.b

    def raw_logit(p_llm):                                # invert the frozen Platt map
        z = np.log(p_llm / (1 - p_llm))
        return (z - cal_b) / cal_a

    base = comp.merge(lab, on="response_id")
    runs = {REFERENCE["tag"]: (base[["response_id", "p_llm"]].assign(z_raw=raw_logit(base.p_llm)), REFERENCE)}
    for f in sorted(args.runs.glob("*.csv")):
        if any(f.stem.startswith(x) for x in args.exclude):
            continue
        if not f.with_suffix(".json").exists():   # not a score_llm_half.py run (e.g. this report's own report.csv)
            continue
        meta = json.loads(f.with_suffix(".json").read_text())
        runs[f.stem] = (pd.read_csv(f), meta)

    ref = runs[REFERENCE["tag"]][0].set_index("response_id")
    rows, common = [], set(base.response_id)
    for df, _ in runs.values():
        common &= set(df.response_id)
    common_ids = sorted(common)

    def view(df, ids):
        d = base.set_index("response_id").loc[ids, ["p_gbm", "is_correct"]].join(
            df.set_index("response_id")[["p_llm", "z_raw"]])
        d["p_ens"] = (1 - W) * d.p_llm + W * d.p_gbm
        return d

    for tag, (df, meta) in runs.items():
        full = view(df, list(df.response_id))
        sub = view(df, common_ids)
        y, ys = full.is_correct.to_numpy(int), sub.is_correct.to_numpy(int)
        # compare each run with the reference on ALL of its own rows (the reference covers the full
        # test set), not only on the rows every run shares
        own = list(full.index)
        dz = (full.z_raw - ref.loc[own, "z_raw"]).to_numpy()
        dp = (full.p_ens - view(ref.reset_index(), own).p_ens).to_numpy()
        llm_ll, llm_auc = score(y, full.p_llm.to_numpy())
        ens_ll, ens_auc = score(y, full.p_ens.to_numpy())
        c_ll, c_auc = score(ys, sub.p_ens.to_numpy())
        cc = str(meta.get("compute_capability", ""))
        rows.append({
            "run": tag, "GPU": meta.get("gpu", ""), "arch": f"{ARCH.get(cc, '?')} (sm {cc})" if cc else "CPU",
            "GPU GB": meta.get("gpu_mem_gb", ""), "driver": meta.get("driver", ""),
            "torch": meta.get("torch", ""), "batch": meta.get("batch_size", ""),
            "rows": len(full), "rows/s": (round(len(full) / meta["score_s"], 2) if meta.get("score_s") else ""),
            "LLM LL": llm_ll, "LLM AUROC": llm_auc, "ens LL": ens_ll, "ens AUROC": ens_auc,
            "ens LL (common rows)": c_ll, "ens AUROC (common rows)": c_auc,
            "z identical to ref %": 100 * np.mean(np.abs(dz) < 1e-4),
            "median |dz|": np.median(np.abs(dz)), "p99 |dz|": np.quantile(np.abs(dz), 0.99),
            "max |dp_ens|": np.max(np.abs(dp)), "sd dp_ens": np.std(dp),
        })
    t = pd.DataFrame(rows)

    lines = ["# GPU noise: the shipped model on different hardware", "",
             f"Released test set; LLM half scored per device with `src/tools/score_llm_half.py`, XGBoost half fixed "
             f"(CPU, identical everywhere), ensemble p = {1 - W:.2f}·p_LLM + {W:.2f}·p_GBM. "
             f"Common rows (scored on every device): {len(common_ids):,}. "
             "`dz` = change in the raw True−False logit vs the reference run; one bf16 step is 0.0625–0.125 "
             "for logits of this size. `rows with z identical to ref`, `|dz|` and `max |Δp|` compare each run "
             "with the reference on all of that run's rows.", "",
             "| run | GPU | arch | GB | batch | rows | LLM LL / AUROC | **ensemble LL / AUROC** | ens LL / AUROC, common rows "
             "| rows with z identical to ref | median / p99 \\|dz\\| | max \\|Δp\\| ens | rows/s |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join([
            r["run"], r["GPU"], r["arch"], str(r["GPU GB"]), str(r["batch"]), f"{r['rows']:,}",
            f"{r['LLM LL']:.5f} / {r['LLM AUROC']:.3f}", f"**{r['ens LL']:.5f} / {r['ens AUROC']:.3f}**",
            f"{r['ens LL (common rows)']:.5f} / {r['ens AUROC (common rows)']:.3f}",
            f"{r['z identical to ref %']:.1f} %", f"{r['median |dz|']:.3f} / {r['p99 |dz|']:.3f}",
            f"{r['max |dp_ens|']:.4f}", str(r["rows/s"])]) + " |")
    full_runs = t[t.rows == t.rows.max()]
    lines += ["", "**Spread across devices** (runs that scored all rows: "
              f"{len(full_runs)}): ensemble log loss range {full_runs['ens LL'].min():.5f}–{full_runs['ens LL'].max():.5f} "
              f"(sd {full_runs['ens LL'].std(ddof=1) if len(full_runs) > 1 else 0:.5f}); AUROC range "
              f"{full_runs['ens AUROC'].min():.3f}–{full_runs['ens AUROC'].max():.3f} "
              f"(sd {full_runs['ens AUROC'].std(ddof=1) if len(full_runs) > 1 else 0:.3f}). "
              f"On the {len(common_ids):,} common rows, all devices: log loss sd "
              f"{t['ens LL (common rows)'].std(ddof=1) if len(t) > 1 else 0:.5f}, AUROC sd "
              f"{t['ens AUROC (common rows)'].std(ddof=1) if len(t) > 1 else 0:.3f}.", ""]
    lines += ["Hardware and software per run:", "",
              "| run | host | CPU | RAM GB | GPUs visible | driver | CUDA (torch) | torch | transformers | attention | peak GPU GB | wall |",
              "|---|---|---|---:|---:|---|---|---|---|---|---:|---:|"]
    for tag, (df, m) in runs.items():
        lines.append("| " + " | ".join(str(x) for x in [
            tag, m.get("host", ""), m.get("cpu", ""), m.get("ram_gb", ""), m.get("visible_gpus", ""),
            m.get("driver", ""), m.get("torch_cuda", ""), m.get("torch", ""), m.get("transformers", ""),
            m.get("attn", ""), m.get("peak_gpu_mem_gb", ""),
            (f"{m['wall_s'] / 3600:.2f} h" if m.get("wall_s") else "")]) + " |")
    out = "\n".join(lines) + "\n"
    print(out)
    dest = args.out or (args.runs / "report.md")
    dest.write_text(out)
    t.to_csv(dest.with_suffix(".csv"), index=False, float_format="%.6f")
    print(f"wrote {dest} and {dest.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
