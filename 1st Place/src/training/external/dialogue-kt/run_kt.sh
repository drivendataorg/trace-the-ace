#!/usr/bin/env bash
# Runs dialogue_kt.main with the environment our workstation needed (run_kt_lsf.sh is the
# LSF cluster variant).
# Usage: ./run_kt.sh <args to dialogue_kt.main>
#   e.g. ./run_kt.sh train --dataset comta --crossval --model_type bkt
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # release: was an absolute cluster path
# The host sets LD_LIBRARY_PATH=/opt/cuda/cuda-12.0/lib64, whose libnvJitLink.so.12
# (CUDA 12.0) lacks the __nvJitLinkAddData_12_1 symbol that torch 2.3's bundled
# cusparse (CUDA 12.1) needs. torch's wheels carry their own CUDA libs via RPATH,
# so clear LD_LIBRARY_PATH to stop the system CUDA from shadowing them.
unset LD_LIBRARY_PATH
export HF_HOME="${HF_HOME:-$ROOT/.hf}"   # release: honour a caller-set HF_HOME (was always $ROOT/.hf)
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # repo's determinism requirement
export CUDA_DEVICE_ORDER=PCI_BUS_ID         # make CUDA indices match `nvidia-smi -L` (else fastest-first)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # release: was 1 (our dev box's A5000); lm.py uses cuda:0
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled                  # no W&B account / offline
# Gated Llama-3.1 needs an HF token. Read the user's existing one at runtime
# (HF_HOME points at the project cache, which has no token of its own). Not hardcoded.
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
cd "$ROOT"
exec "${KT_PYTHON:-$ROOT/.env/bin/python}" -m dialogue_kt.main "$@"   # release: KT_PYTHON overrides the venv
