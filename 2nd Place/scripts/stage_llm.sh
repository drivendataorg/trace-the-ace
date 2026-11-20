#!/usr/bin/env bash
# Assemble one LLM reader bundle: a base model plus LoRA adapters, or a merged dense model.
#
# LoRA bundle (the Qwen3-4B/8B layout: no bundle.json, so the worker's defaults apply):
#   MODEL=Qwen/Qwen3-4B FT_DIR=outputs/models/e13_qwen4 FOLDS=full OUT=<bundle> bash scripts/stage_llm.sh
# Dense Qwen3.5 bundle (what training stage 9 runs; the export must carry its manifest):
#   FAMILY=qwen3_5 ADAPTER_MODE=dense DENSE_DIR=outputs/models/e23_qwen35_9b/full/dense \
#   FOLDS=full EXPECTED_FOLDS="" OUT=outputs/staging/assets/llm_q35 bash scripts/stage_llm.sh
# BASE_MODE=external EXTERNAL_BASE=<dir> skips copying the base, for a runtime that preloads the
# model under huggingface_models/{org}/{model}/ (do not assume Qwen3.5 is preloaded).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
FT_DIR="${FT_DIR:-outputs/models/e13_qwen4b}"
OUT="${OUT:-outputs/submissions/llm_bundle}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# FAMILY=generic (the default) writes no bundle.json, so Qwen3 bundles keep the worker's
# default engine settings.
FAMILY="${FAMILY:-generic}"
ADAPTER_MODE="${ADAPTER_MODE:-lora}"
BASE_MODE="${BASE_MODE:-embedded}"
EXTERNAL_BASE="${EXTERNAL_BASE:-}"
DENSE_DIR="${DENSE_DIR:-}"
EXPECTED_FOLDS="${EXPECTED_FOLDS:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
BUNDLE_NOTE="${BUNDLE_NOTE:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [[ "$ADAPTER_MODE" != "lora" && "$ADAPTER_MODE" != "dense" ]]; then
    echo "!! ADAPTER_MODE must be 'lora' or 'dense' (got '$ADAPTER_MODE')" >&2; exit 2
fi
if [[ "$BASE_MODE" != "embedded" && "$BASE_MODE" != "external" ]]; then
    echo "!! BASE_MODE must be 'embedded' or 'external' (got '$BASE_MODE')" >&2; exit 2
fi
if [[ "$BASE_MODE" == "external" && -z "$EXTERNAL_BASE" ]]; then
    echo "!! BASE_MODE=external requires EXTERNAL_BASE" >&2; exit 2
fi

mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# 1. base
# ---------------------------------------------------------------------------
if [[ "$BASE_MODE" == "external" ]]; then
    echo "== base_mode=external -> not staging weights (runtime supplies $EXTERNAL_BASE)"
elif [[ "$ADAPTER_MODE" == "dense" ]]; then
    # Dense mode: base/ is the merged checkpoint written by scripts/export_qwen35_full_merged.py.
    # Copy it verbatim: re-saving it could change the top-level architecture vLLM 0.21 needs.
    [[ -n "$DENSE_DIR" ]] || { echo "!! ADAPTER_MODE=dense requires DENSE_DIR" >&2; exit 2; }
    [[ -d "$DENSE_DIR" ]] || { echo "!! DENSE_DIR not found: $DENSE_DIR" >&2; exit 2; }
    # A dense directory is not necessarily a finished export: a failed transplant leaves the
    # unmodified base, which loads and serves but is the wrong model. Require the completion
    # manifest the exporter writes only after validation; checked before the copy.
    echo "== validating dense export manifest in $DENSE_DIR"
    DENSE_DIR="$DENSE_DIR" python - <<'PY' || exit 1
import os, sys
sys.path.insert(0, "src")
from ace.llm.dense_manifest import DenseManifestError, require_valid_dense_export
try:
    m = require_valid_dense_export(os.environ["DENSE_DIR"])
except DenseManifestError as e:
    print(f"\n!! refusing to stage this dense checkpoint:\n\n{e}\n", file=sys.stderr)
    sys.exit(1)
print(f"   manifest OK: {m['mapped_tensor_count']} mapped, "
      f"{m['tied_alias_count']} tied alias(es), "
      f"{m.get('untouched_tensor_count', '?')} untouched; "
      f"language_root={m['language_root']}")
PY
    echo "== staging dense checkpoint $DENSE_DIR -> $OUT/base"
    mkdir -p "$OUT/base"
    cp -r "$DENSE_DIR"/. "$OUT/base/"
else
    # materialize the base as a self-contained dir (inference has no internet).
    # Weights + tokenizer only; skips the HF cache symlink indirection.
    echo "== staging base $MODEL -> $OUT/base"
    mkdir -p "$OUT/base"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$OUT/base',
    allow_patterns=['*.safetensors','*.json','*.txt','*.jinja',
                    'tokenizer*','vocab*','merges*','special_tokens*','added_tokens*'])
"
fi

# ---------------------------------------------------------------------------
# 2. adapters (lora mode only; dense bundles deliberately have none)
# ---------------------------------------------------------------------------
STAGED_FOLDS=()
if [[ "$ADAPTER_MODE" == "lora" ]]; then
    mkdir -p "$OUT/adapters"
    for f in $FOLDS; do
        # "full" is the full-fit pseudo-fold (run_training.sh convention): a single
        # deployment adapter trained on every row -> k=1 at inference.
        if [[ "$f" == "full" ]]; then name="full"; else name="fold$f"; fi
        src="$FT_DIR/$name/adapter"
        if [[ ! -s "$src/adapter_model.safetensors" ]]; then
            echo "!! missing adapter: $src -- skipping $name" >&2; continue
        fi
        mkdir -p "$OUT/adapters/$name"
        cp "$src/adapter_config.json" "$src/adapter_model.safetensors" "$OUT/adapters/$name/"
        STAGED_FOLDS+=("$f")
        echo "   staged adapter $name"
    done
    if [[ ${#STAGED_FOLDS[@]} -eq 0 ]]; then
        echo "!! no adapters staged -- refusing to write an empty lora bundle" >&2; exit 1
    fi
else
    # Dense: no adapter dirs, so FOLDS records which fold's adapter was merged into the
    # checkpoint; otherwise the bundle would report every expected fold as missing.
    for f in $FOLDS; do STAGED_FOLDS+=("$f"); done
fi

# ---------------------------------------------------------------------------
# 3. bundle.json, for non-generic families only. Generic (Qwen3) bundles get none, so
#    ace.llm.bundle uses its defaults and the engine kwargs stay those of the shipped readers.
# ---------------------------------------------------------------------------
if [[ "$FAMILY" == "generic" ]]; then
    echo "== family=generic -> no bundle.json (legacy behaviour preserved)"
else
    echo "== writing $OUT/bundle.json (family=$FAMILY, adapter_mode=$ADAPTER_MODE)"
    STAGED_JOINED="${STAGED_FOLDS[*]:-}"
    STAGED_JOINED="$STAGED_JOINED" EXPECTED_FOLDS="$EXPECTED_FOLDS" \
    FAMILY="$FAMILY" ADAPTER_MODE="$ADAPTER_MODE" BASE_MODE="$BASE_MODE" \
    EXTERNAL_BASE="$EXTERNAL_BASE" MAX_LORA_RANK="$MAX_LORA_RANK" \
    BUNDLE_NOTE="$BUNDLE_NOTE" OUT="$OUT" \
    python - <<'PY'
import json, os, sys
sys.path.insert(0, "src")
from ace.llm.bundle import (QWEN35_VLLM_ARCHITECTURE, BundleValidationError,
                            fold_label, parse_meta, validate_bundle)

def folds(env):
    return [fold_label(t) if not t.isdigit() else int(t)
            for t in os.environ.get(env, "").split()]

family = os.environ["FAMILY"]
adapter_mode = os.environ["ADAPTER_MODE"]
meta = {
    "engine": "vllm",
    "architecture_family": family,
    "language_model_only": family == "qwen3_5",
    "adapter_mode": adapter_mode,
    "base_mode": os.environ["BASE_MODE"],
    "dtype": "bfloat16",
    "folds": folds("STAGED_JOINED"),
    "expected_folds": folds("EXPECTED_FOLDS"),
}
if family == "qwen3_5":
    # the only top-level architecture vLLM 0.21 registers for this family
    meta["expected_architecture"] = QWEN35_VLLM_ARCHITECTURE
if adapter_mode == "lora":
    meta["max_lora_rank"] = int(os.environ["MAX_LORA_RANK"])
if os.environ["BASE_MODE"] == "external":
    meta["external_base"] = os.environ["EXTERNAL_BASE"]
if os.environ.get("BUNDLE_NOTE"):
    meta["note"] = os.environ["BUNDLE_NOTE"]

out = os.environ["OUT"]
try:
    parsed = parse_meta(meta, source="stage_llm.sh")   # reject a malformed bundle now
    with open(f"{out}/bundle.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    # resolve_base=False for external: the staging machine does not have the model the
    # runtime preloads. Metadata is still validated (parse_meta requires external_base), and
    # score_worker checks the architecture once the path resolves.
    validate_bundle(out, resolve_base=parsed.base_mode == "embedded")
except BundleValidationError as e:
    # A traceback here would bury the explanation, which is the useful part.
    print(f"\n!! this bundle is NOT deployable — refusing to stage it:\n\n{e}\n",
          file=sys.stderr)
    sys.exit(1)
print(f"   {parsed.describe()}")
if parsed.incomplete_cv:
    print(f"   NOTE: INCOMPLETE CV — missing folds {parsed.missing_folds}; "
          f"this bundle is an exploratory reader, not a cross-validated one")
PY
fi

echo "== bundle sizes =="
SIZE_TARGETS=("$OUT")
[[ -d "$OUT/base" ]] && SIZE_TARGETS+=("$OUT/base")
[[ -d "$OUT/adapters" ]] && SIZE_TARGETS+=("$OUT/adapters")
du -sh "${SIZE_TARGETS[@]}"

n_ad=0
[[ -d "$OUT/adapters" ]] && n_ad=$(find "$OUT/adapters" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "== LLM bundle ready: $OUT  (base + $n_ad adapters, adapter_mode=$ADAPTER_MODE)"
