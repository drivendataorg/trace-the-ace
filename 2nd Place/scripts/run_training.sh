#!/usr/bin/env bash
# Training entry point: raw training data -> every trained asset of the submission.
#
#   DATA_DIR=/path/to/data bash scripts/run_training.sh            # all stages, in order
#   DATA_DIR=/path/to/data bash scripts/run_training.sh 4 5        # selected stages only
#
# Each stage names the environment it needs (envs/setup_envs.sh); the script activates
# .venv-<env> itself. Stages are resumable: a stage whose outputs exist is skipped
# (set FORCE=1 to redo).
#
#  stage  env       what                                                  time (80 GB GPU)
#  -----  --------  ----------------------------------------------------  ----------------
#   0     tabular   download pinned base models                           network only
#   1     tabular   EDA tables (transcript stats, objective overlap)      minutes
#   2     tabular   BGE embeddings: objective coverage + objective vectors minutes (GPU)
#   3     tabular   CV folds (skill clusters) + base blend TF-IDF/LR+GBM  < 1 h (CPU)
#   4     tabular   tabular_v2 features (+ student BGE) and 15 boosters   ~1 h
#   5     llm       Qwen3-4B LoRA: 5 fold adapters + OOF scoring, full    59 h + 25 h
#   6     llm       Qwen3-8B LoRA: 5 fold adapters + OOF scoring, full    82 h + 25 h
#   7     qwen35    Qwen3.5-9B LoRA full fit                              25 h
#   8     tabular   level-2 stack fit                                     seconds
#   9     qwen35/   merge the Qwen3.5 adapter into a vLLM-servable        < 1 h
#         tabular   checkpoint; stage every asset into outputs/staging/
#
# Times are for one 80 GB data-center GPU; an RTX 6000 Pro 96 GB is similar or faster.
# Stages 5-6 produce the out-of-fold predictions the stack (stage 8) is fitted on.
# Set OOF_DIR to point stage 8 at a folder holding previously computed predictions.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_DIR="$(cd "${DATA_DIR:-$ROOT/data}" && pwd)"
TR="$DATA_DIR/train_transcripts"
FORCE="${FORCE:-0}"
OUT=outputs
HF="$ROOT/models/hf"          # pinned base-model snapshots
CACHE="$ROOT/cache"
FOLDS="$OUT/eda/cv_folds_skill.csv"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

use() {  # activate .venv-<name>
  local v="$ROOT/.venv-$1"
  [[ -f "$v/bin/activate" ]] || { echo "!! $v missing: bash envs/setup_envs.sh $1"; exit 2; }
  # shellcheck disable=SC1091
  source "$v/bin/activate"
}
done_or() { [[ "$FORCE" != "1" && -e "$1" ]] && { echo "   skip (exists: $1)"; return 0; }; return 1; }
banner() { echo; echo "================ stage $1: $2"; }

# Recipe shared by the three LoRA readers (identical except the base model).
LORA_ARGS=(--serialize 1 --max-input-tokens 8192 --epochs 1 --lr 1e-4
           --lora-r 16 --lora-alpha 32 --batch 1 --accum 16 --seed 42)

stage0() { banner 0 "pinned base models -> $HF"; use tabular
  python - "$HF" <<'PY'
import sys
from huggingface_hub import snapshot_download
root = sys.argv[1]
pins = {  # exact revisions (the current ones on the Hub; byte-identical to the shipped files)
    "Qwen/Qwen3-4B":         "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B":         "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3.5-9B":       "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
}
for repo, rev in pins.items():
    d = f"{root}/{repo.split('/')[1]}"
    snapshot_download(repo, revision=rev, local_dir=d,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja",
                                      "tokenizer*", "vocab*", "merges*", "1_Pooling/*"])
    print("ok", repo, rev, "->", d)
PY
}

stage1() { banner 1 "EDA tables"; use tabular
  done_or "$OUT/eda/transcript_stats.csv" && return
  python src/ace/eda/run_eda.py --data-dir "$DATA_DIR" --transcripts-dir "$TR" --out "$OUT/eda"
}

stage2() { banner 2 "BGE embeddings"; use tabular
  done_or "$CACHE/emb_objectives.npy" && return
  python src/ace/eda/run_embeddings.py --data-dir "$DATA_DIR" --transcripts-dir "$TR" \
    --eda-dir "$OUT/eda" --out "$OUT/embeddings" --cache-dir "$CACHE" --model "$HF/bge-base-en-v1.5"
}

stage3() { banner 3 "skill-cluster CV folds + base blend"; use tabular
  done_or "$OUT/models/sub5_fit/oof.csv" && return
  # writes $FOLDS (skill-cluster folds, seed 42), the base-blend assets into outputs/staging/assets
  # (tfidf_*.npz, logreg.npz, gbm/, te_objective.npz, topic_map.json, config.json, bge export)
  python -m ace.submission.fit_final_v2 --data-dir "$DATA_DIR" --transcripts-dir "$TR" \
    --eda-dir "$OUT/eda" --emb-dir "$OUT/embeddings" --cache-dir "$CACHE" \
    --staging "$OUT/staging" --report-out "$OUT/models/sub5_fit" \
    --seeds 42,1337,2024 --group-threshold 0.90 --bge-model "$HF/bge-base-en-v1.5"
}

stage4() { banner 4 "tabular_v2"; use tabular
  local fd="$OUT/features/tabular_v2" md="$OUT/models/tabular_v2_bge_no_telo"
  if ! done_or "$md/oof.csv"; then
    python -m ace.models.build_tabular_v2_features --data-dir "$DATA_DIR" --transcripts-dir "$TR" --out "$fd"
    python -m ace.models.build_tabular_v2_features --data-dir "$DATA_DIR" --transcripts-dir "$TR" --out "$fd" \
      --with-bge --bge-model "$HF/bge-base-en-v1.5"
    python -m ace.models.train_tabular_v2 --data-dir "$DATA_DIR" --eda-dir "$OUT/eda" \
      --emb-dir "$OUT/embeddings" --cache-dir "$CACHE" --feature-dir "$fd" --folds "$FOLDS" \
      --out "$md" --seeds 42,1337,2024 --num-rounds 5000 --early-stop 250 --no-te-lo \
      --threads 16   # as in the original run; recorded in each booster file (byte-identical --check)
  fi
  python scripts/stage_tabular_v2.py --model-dir "$md" --out "$OUT/staging/assets/tabular_v2"
}

lora_reader() {  # $1 base dir  $2 out name ; trains folds 0-4 (+OOF scoring) and the full fit
  local base="$1" od="$OUT/models/$2"
  for f in 0 1 2 3 4 full; do
    local name; [[ "$f" == full ]] && name=full || name="fold$f"
    local fa; [[ "$f" == full ]] && fa=-1 || fa="$f"
    if ! done_or "$od/$name/config.json"; then
      python -m ace.llm.train_lora --model "$base" --data-dir "$DATA_DIR" --transcripts-dir "$TR" \
        --folds "$FOLDS" --fold "$fa" --out "$od/$name" "${LORA_ARGS[@]}"
    fi
    if [[ "$f" != full ]] && ! done_or "$od/p_fold$f.csv"; then
      python -m ace.llm.score_vllm --model "$base" --adapter "$od/$name/adapter" \
        --data-dir "$DATA_DIR" --transcripts-dir "$TR" --folds "$FOLDS" --fold "$f" \
        --serialize 1 --out "$od/p_fold$f.csv"
    fi
  done
}

stage5() { banner 5 "Qwen3-4B LoRA reader"; use llm; lora_reader "$HF/Qwen3-4B" e13_qwen4; }
stage6() { banner 6 "Qwen3-8B LoRA reader"; use llm; lora_reader "$HF/Qwen3-8B" e13_qwen8b; }

stage7() { banner 7 "Qwen3.5-9B LoRA reader (full fit)"; use qwen35
  local od="$OUT/models/e23_qwen35_9b"
  done_or "$od/full/config.json" && return
  # Qwen3.5 enters the stack only through the fixed manual blend (w=0.50), so only the
  # deployment full fit is required. Fold adapters (same command with --fold 0..4, scored
  # with ace.llm.score_hf) were trained for analysis only.
  python -m ace.llm.train_lora --model "$HF/Qwen3.5-9B" --data-dir "$DATA_DIR" --transcripts-dir "$TR" \
    --folds "$FOLDS" --fold -1 --out "$od/full" --prompt-mode standard "${LORA_ARGS[@]}"
}

stage8() { banner 8 "level-2 stack"; use tabular
  # Out-of-fold predictions from stages 3-6 (or OOF_DIR with the same layout).
  local models="${OOF_DIR:-$OUT/models}"
  for f in sub5_fit/oof.csv tabular_v2_bge_no_telo/oof.csv e13_qwen4/p_fold4.csv e13_qwen8b/p_fold4.csv; do
    [[ -s "$models/$f" ]] || { echo "!! $models/$f missing: run stages 3-6 first"; exit 2; }
  done
  echo "   OOF source: $models"
  # fit: any input or write error stops training here (set -e)
  python scripts/fit_stack_sub9.py --models "$models" --out "$OUT/staging/assets/llm_stack.json"
  # compare with the shipped W050 stack: fails only if the method (structure) changed;
  # coefficient differences are reported, and are expected after a retrain
  python scripts/fit_stack_sub9.py --compare "$OUT/staging/assets/llm_stack.json" \
    inference/assets/llm_stack.json
}

stage9() { banner 9 "assemble outputs/staging"; local S="$OUT/staging/assets"
  use qwen35
  if ! done_or "$OUT/models/e23_qwen35_9b/full/dense/model.safetensors.index.json"; then
    python scripts/export_qwen35_full_merged.py --base "$HF/Qwen3.5-9B" \
      --adapter "$OUT/models/e23_qwen35_9b/full/adapter" --out "$OUT/models/e23_qwen35_9b/full/dense"
  fi
  use tabular
  FAMILY=qwen3_5 ADAPTER_MODE=dense DENSE_DIR="$OUT/models/e23_qwen35_9b/full/dense" \
    FOLDS=full EXPECTED_FOLDS="" OUT="$S/llm_q35" bash scripts/stage_llm.sh
  # Qwen3 bundles (no bundle.json, worker defaults): pinned base snapshot + the full-fit adapter
  for pair in "llm_4b:Qwen3-4B:e13_qwen4" "llm_8b:Qwen3-8B:e13_qwen8b"; do
    IFS=: read -r dir base run <<< "$pair"
    mkdir -p "$S/$dir/base" "$S/$dir/adapters/full"
    cp "$HF/$base"/*.json "$HF/$base"/*.safetensors "$HF/$base"/merges.txt "$S/$dir/base/"
    cp "$OUT/models/$run/full/adapter/adapter_config.json" \
       "$OUT/models/$run/full/adapter/adapter_model.safetensors" "$S/$dir/adapters/full/"
    echo "   staged $dir"
  done
  mkdir -p "$S/ace" && cp -r src/ace/{__init__.py,features.py,transcripts.py,utils.py,tabular_v2_features.py} "$S/ace/"
  mkdir -p "$S/ace/llm" && cp src/ace/llm/{__init__.py,bundle.py,model_family.py,objective_context.py,prepare_objective_contexts.py,runtime_check.py,score_worker.py,serialize.py} "$S/ace/llm/"
  cp src/ace/submission/main.py "$OUT/staging/main.py"
  echo "   staged tree: $OUT/staging  (same layout as inference/); run it with:"
  echo "   INFERENCE_DIR=$OUT/staging DATA_DIR=data bash scripts/run_inference.sh"
}

STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(0 1 2 3 4 5 6 7 8 9)
for s in "${STAGES[@]}"; do "stage$s"; done
echo; echo "== training stages done: ${STAGES[*]}"
