#!/usr/bin/env bash
# LSF (DTU DCC) variant of run_kt.sh: runs dialogue_kt.main with the cluster environment.
# Usage: ./run_kt_lsf.sh <args to dialogue_kt.main>
# Differences from run_kt.sh: it never sets CUDA_VISIBLE_DEVICES (run_kt.sh defaults it
# to 0), it runs Hugging Face offline,
# and it sets PYTORCH_CUDA_ALLOC_CONF. Under LSF `-gpu mode=exclusive_process` the
# scheduler exports CUDA_VISIBLE_DEVICES for the allocated GPU, and dialogue_kt/utils.py
# uses the default CUDA device (cuda:0), so the job runs on its own GPU. Overriding the
# variable here would break that.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # release: was an absolute cluster path
unset LD_LIBRARY_PATH                       # stop system CUDA 12.0 shadowing torch's bundled 12.1
export HF_HOME="${HF_HOME:-$ROOT/.hf}"   # release: honour a caller-set HF_HOME (was always $ROOT/.hf)
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # repo's determinism requirement
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOMs (bf16 long-seq logit spikes)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1                     # compute nodes have no internet; weights are pre-cached
export TRANSFORMERS_OFFLINE=1
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
cd "$ROOT"
echo "[run_kt_lsf] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
exec "${KT_PYTHON:-$ROOT/.env/bin/python}" -m dialogue_kt.main "$@"   # release: KT_PYTHON overrides the venv
