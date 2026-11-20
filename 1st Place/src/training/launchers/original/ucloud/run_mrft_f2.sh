#!/bin/bash
cd /work/magnificat/dialogue-kt
export HF_HOME=/work/magnificat/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LLMKT_MAX_SEQ_LEN=8192 CUDA_VISIBLE_DEVICES=0
PY=/work/magnificat/venv/bin/python
echo "[1/3] MRBench-tune fold-2 $(date)"
TRACEACE_PKL=data/annotated/mrbench_train.pkl \
  $PY -m dialogue_kt.main train --dataset traceace --model_type lmkt --train_all \
  --pt_model_name lmkt_traceace_qwen3_8b_full_2 \
  --pack_kcs 0 --quantize 0 --inc_first_label --epochs 1 --lr 1e-4 \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f2
echo "MRFT_TRAIN_DONE $(date)"
ln -sfn lmkt_mrft_sa_8b_a1f2 saved_models/lmkt_mrft_sa_8b_a1f2_2
ln -sfn lmkt_mrft_sa_8b_a1f2 saved_models/lmkt_mrft_sa_8b_a1f2_1
echo "[2/3] eval TtA fold-2 $(date)"
TRACEACE_PKL=data/annotated/traceace_full.pkl \
  $PY -m dialogue_kt.main test --dataset traceace --model_type lmkt --fold 2 \
  --batch_size 8 --pack_kcs 0 --quantize 0 --inc_first_label \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f2
echo "TTA_F2_EVAL_DONE $(date)"
echo "[3/3] eval QATD $(date)"
TRACEACE_EVAL_ALL=1 TRACEACE_PKL=data/annotated/qatd_test.pkl \
  $PY -m dialogue_kt.main test --dataset traceace --model_type lmkt --fold 1 \
  --batch_size 8 --pack_kcs 0 --quantize 0 --inc_first_label \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f2
echo "MRFT_F2_ALL_DONE $(date)"
