#!/bin/bash
# Step 02a: full-corpus TtA correctness annotation on an 8x B200 job: 4 parallel Qwen3.5-122B-FP8
# instances (TP=2), each annotating shard i/4. Resume-safe: rerunning skips
# sessions already in each shard csv. Merge + dedupe when all four finish.
cd /work/magnificat/dialogue-kt
source /work/magnificat/vllm_env.sh
for i in 0 1 2 3; do
  g0=$((i*2)); g1=$((i*2+1))
  nohup env CUDA_VISIBLE_DEVICES=$g0,$g1 MAGNIFICAT=/work/magnificat/repo \
    ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 \
    ANNO_MAXTOK=4096 ANNO_CHUNK_TURNS=10 ANNO_REF=results/NONEXISTENT.csv ANNO_SHARD=$i/4 \
    /work/magnificat/venv-vllm/bin/python annotate_traceace_vllm.py \
    > /work/magnificat/anno_prod_shard$i.log 2>&1 &
  echo "shard $i/4 on GPUs $g0,$g1 pid $!"
done
