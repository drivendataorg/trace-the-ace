#!/usr/bin/env bash
# Inference entry point: raw test data + model weights -> submission.csv.
#
# Runs inference/main.py exactly as the competition runtime did (same code, same assets,
# verified byte-identical to the scored archive). Nothing is retrained.
#
#   source .venv-inference/bin/activate
#   DATA_DIR=/path/to/data bash scripts/run_inference.sh
#   INFERENCE_DIR=outputs/staging DATA_DIR=data bash scripts/run_inference.sh   # a retrained tree
#
# INFERENCE_DIR (default: inference/) is the folder main.py runs from. Only the default one is
# checked against the scored archive (size + CRC32); any other folder holds retrained weights,
# so only its main.py and assets/llm_stack.json are required.
#
# DATA_DIR must contain (competition format):
#   test_features.csv  submission_format.csv  test_transcripts/<session_id>.csv
# Output: submission/submission.csv (+ submission/log.txt)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data}" && pwd)"
# absolute before the cd into the inference folder below, so relative paths never land there
mkdir -p "${OUT_DIR:-$ROOT/submission}"
OUT_DIR="$(cd "${OUT_DIR:-$ROOT/submission}" && pwd)"
INFERENCE_DIR="$(cd "${INFERENCE_DIR:-$ROOT/inference}" && pwd)"
SHIPPED="$(cd "$ROOT/inference" && pwd)"

for f in test_features.csv submission_format.csv; do
  [[ -s "$DATA_DIR/$f" ]] || { echo "!! $DATA_DIR/$f not found"; exit 2; }
done
[[ -d "$DATA_DIR/test_transcripts" ]] || { echo "!! $DATA_DIR/test_transcripts/ not found"; exit 2; }

if [[ "$INFERENCE_DIR" == "$SHIPPED" ]]; then
  echo "== verifying inference tree against the scored archive (size + CRC32)"
  python "$ROOT/scripts/download_weights.py" --verify-only > "$ROOT/.verify.log" \
    || { tail -20 "$ROOT/.verify.log"; echo "!! run scripts/download_weights.py first"; exit 1; }
  tail -1 "$ROOT/.verify.log"
else
  echo "== INFERENCE_DIR=$INFERENCE_DIR: retrained tree, skipping the scored-archive weights check"
  for f in main.py assets/llm_stack.json; do
    [[ -s "$INFERENCE_DIR/$f" ]] || { echo "!! $INFERENCE_DIR/$f not found"; exit 2; }
  done
fi

# main.py reads ./data and writes ./submission.csv relative to its own folder, like the runtime.
# `data` is a symlink so the 44 GB tree is never copied.
cd "$INFERENCE_DIR"
rm -f data submission.csv
ln -s "$DATA_DIR" data
trap 'rm -f "$INFERENCE_DIR/data"' EXIT

mkdir -p "$OUT_DIR"
start=$(date +%s)
# Same environment the runtime gives the job: offline, strict reader policy.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python main.py 2>&1 | tee "$OUT_DIR/log.txt"
mv submission.csv "$OUT_DIR/submission.csv"
echo "== done in $(( ($(date +%s) - start) / 60 )) min -> $OUT_DIR/submission.csv"
