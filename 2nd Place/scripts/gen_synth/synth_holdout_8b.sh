#!/usr/bin/env bash
# Synthetic-holdout logit-scale check for one reader, end to end from adapter/ only.
#
# The stack was fitted on fold-adapter OOF logits but ships the full-fit adapter at k=1;
# only the synthetic holdout is unseen by both (see scripts/compare_synth_holdout.py).
# Per tag (default: full fold0 fold4):
#   1. rebuild <tag>/merged from <tag>/adapter (the peft merge is exact),
#   2. score it on the synthetic rows -> outputs/models/<OUT_NAME>/synth/p_<tag>.csv,
#   3. delete <tag>/merged (~16 GB) unless KEEP_MERGED=1,
# then compare_synth_holdout.py reports the full-fit vs fold logit-std ratio. Resumable: an
# existing p_<tag>.csv skips that tag. Model id and serialize version come from each tag's
# config.json. Needs peft and vLLM (the llm env, envs/requirements-llm.txt).
#
#     bash scripts/gen_synth/synth_holdout_8b.sh [full fold0 fold1 fold2 fold3 fold4]
#     OUT_NAME=e13_qwen4 bash scripts/gen_synth/synth_holdout_8b.sh      # the Qwen3-4B reader
set -euo pipefail

# optional machine-local settings (HF_TOKEN, HF_HUB_DISABLE_XET) in scripts/env.sh; git-ignored
[[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

OUT_NAME="${OUT_NAME:-e13_qwen8b}"
OUT_DIR="$REPO/outputs/models/$OUT_NAME"
SYNTH_DIR="${SYNTH_DIR:-$REPO/outputs/synth/pilot}"
STACK="${STACK:-$REPO/outputs/models/sub7_8b_fit/llm_stack.json}"
SYNTH_OUT="$OUT_DIR/synth"
KEEP_MERGED="${KEEP_MERGED:-0}"
# vLLM GPU memory fraction: 0.90 for a solo run; lower it to share the GPU (on 80 GB,
# GPU_MEM_UTIL=0.5 leaves ~40 GB for the 8B: 16 GB weights + KV cache). Only the scoring
# step uses the GPU; the merge runs on CPU/RAM.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
if [[ $# -gt 0 ]]; then TAGS=("$@"); else TAGS=(full fold0 fold4); fi

mkdir -p "$SYNTH_OUT"
echo "== synth-holdout: out=$OUT_DIR synth=$SYNTH_DIR tags=${TAGS[*]} keep_merged=$KEEP_MERGED"

# ---- filename shim: score_vllm wants train_features.csv / train_labels.csv;
#      synth has features.csv / labels.csv. --fold -1 scores every synth row.
SHIM="$(mktemp -d)"
trap 'rm -rf "$SHIM"' EXIT
ln -sf "$SYNTH_DIR/features.csv" "$SHIM/train_features.csv"
ln -sf "$SYNTH_DIR/labels.csv"   "$SHIM/train_labels.csv"

for tag in "${TAGS[@]}"; do
    d="$OUT_DIR/$tag"
    cfg="$d/config.json"
    out_csv="$SYNTH_OUT/p_${tag}.csv"

    if [[ -s "$out_csv" ]]; then
        echo "== $tag: $out_csv exists — skipping (merge+score)"
        continue
    fi
    [[ -s "$cfg" ]] || { echo "!! $cfg missing — need $tag/config.json + $tag/adapter"; exit 1; }
    [[ -s "$d/adapter/adapter_model.safetensors" ]] || { echo "!! $d/adapter missing"; exit 1; }

    MODEL="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['model'])" "$cfg")"
    # the first 4B fold runs predate the serialize key in config.json; all these runs
    # use serialize v1, so default to 1 when the key is absent
    SER="$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('serialize', 1))" "$cfg")"

    # ---- 1. rebuild merged/ from adapter/ (exact; marker: merged/config.json)
    if [[ -s "$d/merged/config.json" ]]; then
        echo "== $tag: merged/ already present — reusing"
    else
        echo "== $tag: merging $MODEL + adapter -> $d/merged"
        python - "$MODEL" "$d/adapter" "$d/merged" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base, adir, mdir = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, adir).merge_and_unload()
m.save_pretrained(mdir, safe_serialization=True)
AutoTokenizer.from_pretrained(adir).save_pretrained(mdir)
print("[merge] wrote", mdir)
PY
    fi

    # ---- 2. score the merged model (no --adapter: avoids a LoRA bug in vLLM 0.8.4)
    echo "== $tag: scoring synth ($(wc -l < "$SYNTH_DIR/features.csv") rows, serialize=$SER) -> $out_csv"
    python -m ace.llm.score_vllm \
        --model "$d/merged" \
        --data-dir "$SHIM" \
        --transcripts-dir "$SYNTH_DIR/transcripts" \
        --fold -1 --serialize "$SER" \
        --gpu-mem-util "$GPU_MEM_UTIL" \
        --out "$out_csv" 2>&1 | tee "$SYNTH_OUT/score_${tag}.log"

    # ---- 3. reclaim disk unless asked to keep
    if [[ "$KEEP_MERGED" != "1" ]]; then
        echo "== $tag: removing $d/merged (~16 GB)"; rm -rf "$d/merged"
    fi
done

# ---- verdict: full-fit vs fold logit-scale (needs p_full.csv + >=1 fold csv)
full_csv="$SYNTH_OUT/p_full.csv"
fold_csvs=("$SYNTH_OUT"/p_fold*.csv)
if [[ -s "$full_csv" && -s "${fold_csvs[0]:-/nonexistent}" ]]; then
    echo "== compare: full-fit vs ${#fold_csvs[@]} fold adapter(s)"
    stack_arg=(); [[ -s "$STACK" ]] && stack_arg=(--stack "$STACK")
    python "$REPO/scripts/compare_synth_holdout.py" \
        --full "$full_csv" --folds "${fold_csvs[@]}" \
        --meta "$SYNTH_DIR/meta.jsonl" \
        --out "$SYNTH_OUT/synth_holdout.md" "${stack_arg[@]}"
else
    echo "!! need p_full.csv AND at least one p_fold*.csv to compare "
    echo "   (have: $(ls "$SYNTH_OUT"/p_*.csv 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '))"
    echo "   rerun including 'full' and a fold, e.g.: bash scripts/gen_synth/synth_holdout_8b.sh full fold0"
    exit 1
fi
echo "== done -> $SYNTH_OUT/synth_holdout.md"
