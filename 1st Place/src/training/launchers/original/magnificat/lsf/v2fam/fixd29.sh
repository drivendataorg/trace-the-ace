#!/usr/bin/env bash
# Steps 05a and 05b: train the XGBoost model, then score its OOF predictions (rescore_clean.py).
# Columns: 152 (including the 34 d29dst_ columns, --extra-csv d29dst_=...) - 21 (ntodst_,
# --drop-cols) + 46 (v2) = 177.
#
# The extra blocks are ablated in `rate-only` mode (--extra-no-ablate), not `soft`; see the
# trainer's --extra-ablate-mode help.
set -uo pipefail
cd /dtu-compute/sciofsci/magnificat
export OMP_NUM_THREADS=1
ARM=$1; PRE=$2; CSV=$3
echo "host=$(hostname) arm=$ARM start=$(date -Is)"
artifacts/rtgpu312/bin/python scripts/train_model_e3l.py \
    --n-seeds 5 --n-jobs 12 --par 4 --no-loknn --swap-student --extra-no-ablate \
    --drop-cols ntodst_ \
    --extra-csv "d29dst_=artifacts/doc29_features_distilled.csv" \
    --extra-csv "${PRE}=${CSV}" \
    --out "artifacts/baseline_${ARM}.joblib" \
    --table-out "artifacts/loknn_table_${ARM}.joblib" \
    --oof-out "artifacts/${ARM}_oof.csv"
rc=$?; echo "TRAIN-RC=$rc"; [ $rc -eq 0 ] || exit $rc
OMP_NUM_THREADS=4 artifacts/rtgpu312/bin/python scripts/rescore_clean.py \
    --ref "ref=artifacts/lokey_ref_oof.csv" \
    --cand "${ARM}=artifacts/${ARM}_oof.csv" --tag "_${ARM}"
echo "FIXD29-$ARM-DONE rc=$? $(date -Is)"
