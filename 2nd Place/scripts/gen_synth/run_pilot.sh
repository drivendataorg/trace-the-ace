#!/usr/bin/env bash
# Synthetic holdout: objectives -> latents -> transcripts -> decontamination -> noise -> QC.
# Resumable: generate.py appends, and finished stages are skipped.
#
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 N=300 bash scripts/gen_synth/run_pilot.sh
#   FRESH=1 ...   regenerate latents and transcripts and re-measure the noise profile
#   REGEN=1 ...   regenerate transcripts and re-measure the profile, keep sessions.jsonl
# objectives.csv is kept in both cases.
set -euo pipefail

# optional machine-local settings (e.g. HF_TOKEN, HF hub quirks) in scripts/env.sh;
# git-ignored, never shipped
[[ -f "$(dirname "${BASH_SOURCE[0]}")/../env.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
N="${N:-300}"
DATA_DIR="${DATA_DIR:-data}"
TRANSCRIPTS_DIR="${TRANSCRIPTS_DIR:-$DATA_DIR/train_transcripts}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SYNTH="$REPO/outputs/synth"
PROFILE="$REPO/outputs/eda/noise_profile.json"
cd "$REPO"

if [[ "${FRESH:-0}" == "1" ]]; then
    echo "== FRESH=1: wiping sessions/clean/pilot + noise profile (objectives kept)"
    rm -f "$SYNTH/sessions.jsonl" "$SYNTH/clean.jsonl" "$PROFILE"
    rm -rf "$SYNTH/pilot"
elif [[ "${REGEN:-0}" == "1" ]]; then
    echo "== REGEN=1: wiping clean/pilot + noise profile (sessions + objectives kept)"
    rm -f "$SYNTH/clean.jsonl" "$PROFILE"
    rm -rf "$SYNTH/pilot"
fi

if [[ ! -s "$PROFILE" ]]; then
    python scripts/gen_synth/noise_profile.py \
        --transcripts-dir "$TRANSCRIPTS_DIR" --data-dir "$DATA_DIR" --out "$PROFILE"
fi

if [[ ! -s "$SYNTH/objectives.csv" ]]; then
    python scripts/gen_synth/make_objectives.py --model "$MODEL" \
        --data-dir "$DATA_DIR" --out "$SYNTH/objectives.csv"
fi

if [[ ! -s "$SYNTH/sessions.jsonl" ]]; then
    python scripts/gen_synth/sample_latents.py --objectives "$SYNTH/objectives.csv" \
        --profile "$PROFILE" --n "$N" --seed "${SEED:-42}" --out "$SYNTH/sessions.jsonl"
fi

python scripts/gen_synth/generate.py --model "$MODEL" \
    --sessions "$SYNTH/sessions.jsonl" --out "$SYNTH/clean.jsonl" \
    --exemplar-dir "$TRANSCRIPTS_DIR"

# remove verbatim copies of the real exemplar (repairs clean.jsonl in place, using the
# exemplar_src field written by generate.py)
python scripts/gen_synth/decontaminate.py --clean "$SYNTH/clean.jsonl" \
    --transcripts-dir "$TRANSCRIPTS_DIR"

python scripts/gen_synth/inject_noise.py --clean "$SYNTH/clean.jsonl" \
    --sessions "$SYNTH/sessions.jsonl" --profile "$PROFILE" --out-dir "$SYNTH/pilot"

python scripts/gen_synth/qc.py --synth-dir "$SYNTH/pilot" \
    --real-stats "$REPO/outputs/eda/transcript_stats.csv" \
    --transcripts-dir "$TRANSCRIPTS_DIR"

echo "== pilot done: $SYNTH/pilot (QC.md inside)"
