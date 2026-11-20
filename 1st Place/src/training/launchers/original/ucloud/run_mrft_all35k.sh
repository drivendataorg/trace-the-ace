#!/bin/bash
set -e
cd /work/magnificat/dialogue-kt
export HF_HOME=/work/magnificat/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LLMKT_MAX_SEQ_LEN=8192 CUDA_VISIBLE_DEVICES=0
PY=/work/magnificat/venv/bin/python
echo "[0/3] QATD baseline: all-35k A1 parent $(date)"
TRACEACE_EVAL_ALL=1 TRACEACE_PKL=data/annotated/qatd_test.pkl \
  $PY -m dialogue_kt.main test --dataset traceace --model_type lmkt --fold 1 \
  --batch_size 8 --pack_kcs 0 --quantize 0 --inc_first_label \
  --base_model Qwen/Qwen3-8B --model_name lmkt_traceace_qwen3_8b_full_final
echo "BASELINE_QATD_DONE $(date)"
echo "[1/3] MRBench-tune all-35k (pt=full_final, 1ep lr1e-4) $(date)"
TRACEACE_PKL=data/annotated/mrbench_train.pkl \
  $PY -m dialogue_kt.main train --dataset traceace --model_type lmkt --train_all \
  --pt_model_name lmkt_traceace_qwen3_8b_full_final \
  --pack_kcs 0 --quantize 0 --inc_first_label --epochs 1 --lr 1e-4 \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_all35k
echo "MRFT_ALL35K_TRAIN_DONE $(date)"
ln -sfn lmkt_mrft_sa_8b_all35k saved_models/lmkt_mrft_sa_8b_all35k_1
echo "[2/3] QATD treatment: MRBench-tuned all-35k $(date)"
TRACEACE_EVAL_ALL=1 TRACEACE_PKL=data/annotated/qatd_test.pkl \
  $PY -m dialogue_kt.main test --dataset traceace --model_type lmkt --fold 1 \
  --batch_size 8 --pack_kcs 0 --quantize 0 --inc_first_label \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_all35k
echo "MRFT_ALL35K_ALL_DONE $(date)"
