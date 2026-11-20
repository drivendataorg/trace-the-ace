#!/usr/bin/env bash
# Verify a downloaded Qwen/Qwen3-8B against the exact base weights the shipped model was built and
# scored on (HF revision b968826d9c46dd6066d109eabc6255188de91218).
#
#   bash src/tools/check_base_model.sh                  # checks models/huggingface_models/Qwen/Qwen3-8B
#   bash src/tools/check_base_model.sh /path/to/Qwen3-8B
#
# Prints one line per file (7 files, from qwen3_8b_base_sha256.txt next to this script); every line
# must say OK, and the exit status is non-zero if a file is missing or differs. A different revision
# of the base would still load, but the LoRA delta was trained on these weights, so predictions
# would move.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=${1:-"$HERE/../../models/huggingface_models/Qwen/Qwen3-8B"}
cd "$DIR" && sha256sum -c "$HERE/qwen3_8b_base_sha256.txt"
