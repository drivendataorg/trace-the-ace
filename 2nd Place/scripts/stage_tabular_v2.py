#!/usr/bin/env python
r"""Stage a trained tabular_v2 model for inference (assets/tabular_v2/).

Copies the 15 early-stopped CV boosters (5 folds x 3 seeds) and writes the config main.py reads.
The 3 fixed-round full-fit boosters are not shipped, and logit_scale stays 1.0: no holdout
exists that all 15 boosters were held out from, so the ensemble scale cannot be measured.
With the original training outputs this reproduces inference/assets/tabular_v2/ byte for byte:

    python scripts/stage_tabular_v2.py --model-dir outputs/models/tabular_v2_bge_no_telo \
        --out outputs/staging/assets/tabular_v2 --check inference/assets/tabular_v2
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

NOTE = ("Ships the 15 early-stopped CV boosters (5 folds x 3 seeds). The 3 fixed-round full-fit "
        "boosters are deliberately NOT shipped. logit_scale 1.0 = uncorrected: the "
        "ensemble-vs-single-booster logit scale is not cleanly measurable (no holdout exists that "
        "all 15 were held out from) and no correction is inferred from test data. Inference "
        "averages PROBABILITIES across the 15 boosters.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/models/tabular_v2_bge_no_telo")
    ap.add_argument("--out", default="outputs/staging/assets/tabular_v2")
    ap.add_argument("--check", default=None, help="reference dir to compare against")
    a = ap.parse_args()
    src, out = Path(a.model_dir), Path(a.out)

    train_cfg = json.loads((src / "config.json").read_text())
    feats = list(train_cfg["features"])
    if "te_lo" in feats:
        raise SystemExit("this is not the te_lo-free fit (train with --no-te-lo)")
    models = sorted(p.name for p in (src / "models").glob("seed*_fold*.txt"))
    if len(models) != 15:
        raise SystemExit(f"expected 15 CV boosters, found {len(models)}")

    (out / "models").mkdir(parents=True, exist_ok=True)
    for m in models:
        shutil.copyfile(src / "models" / m, out / "models" / m)
    cfg = {"version": 1, "model": "tabular_v2_cv15", "features": feats,
           "n_features": len(feats), "n_models": len(models), "models": models,
           "logit_scale": 1.0,
           "source": "outputs/models/tabular_v2_bge_no_telo (BGE, te_lo-free, "
                     "early-stopped 5-fold x 3-seed)",
           "note": NOTE}
    # CRLF line endings, as in the shipped file, so --check is byte-exact
    (out / "config.json").write_bytes(json.dumps(cfg, indent=2).replace("\n", "\r\n").encode())
    print(f"staged {len(models)} boosters + config ({len(feats)} features) -> {out}")

    if a.check:
        ref = Path(a.check)
        names = ["config.json"] + [f"models/{m}" for m in models]
        diff = [n for n in names if not filecmp.cmp(out / n, ref / n, shallow=False)]
        if diff:
            print("MISMATCH vs reference:", diff)
            return 1
        print(f"OK: byte-identical to {ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
