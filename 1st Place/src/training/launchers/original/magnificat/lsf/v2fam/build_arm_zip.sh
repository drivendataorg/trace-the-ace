#!/usr/bin/env bash
# Build a submission zip that differs from the SHIPPED v2shrunkD archive by exactly one file:
# assets/baseline.joblib. Built from the archive zip's bytes, not from a working tree.
# Arg: the XGBoost model name (artifacts/baseline_<name>.joblib).
set -euo pipefail
cd /dtu-compute/sciofsci/magnificat
ARM=$1
SRC=submission_archive/2026-08-27_1825_ens-v2shrunkD-keephdr-w043.zip
W=$(mktemp -d); D=$W/z
unzip -q "$SRC" -d "$D"
cp -f "artifacts/baseline_${ARM}.joblib" "$D/assets/baseline.joblib"
find "$D" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
( cd "$D/assets" && rm -f manifest_sha256.txt && \
  find . -type f ! -name manifest_sha256.txt -printf '%P\n' | LC_ALL=C sort | \
  xargs -d '\n' sha256sum > manifest_sha256.txt )
# structural gates (no GPU): CRC, entry count, leak scan, one-variable diff
Z=$(pwd)/artifacts/ens-${ARM}-keephdr-w043.zip; rm -f "$Z"
( cd "$D" && zip -q -r -X "$Z" main.py main_llmkt_lora.py assets -x '*__pycache__*' '*.pyc' )
unzip -t "$Z" >/dev/null && echo "CRC OK" || { echo "CRC FAIL"; exit 1; }
for m in egate_infer qtremb_infer hp2b3c misfit seq_features dq_features loknn_table; do
  n=$(unzip -l "$Z" | grep -c "$m" || true); [ "$n" -eq 0 ] || { echo "LEAK $m=$n"; exit 1; }
done
echo "leak scan clean"
echo "entries: $(unzip -l "$Z" | tail -1)  (base: $(unzip -l "$SRC" | tail -1))"
echo "w=$(unzip -p "$Z" assets/blend_weight.txt)  keephdr=$(unzip -p "$Z" main.py | grep -c LLMKT_KEEP_HEADER)"
R=$W/r; mkdir -p "$R"; unzip -q "$SRC" -d "$R"
same=0; diff=0
( cd "$R" && find . -type f ! -name manifest_sha256.txt -printf '%P\n' | LC_ALL=C sort ) > $W/lst
while read -r f; do cmp -s "$R/$f" "$D/$f" && same=$((same+1)) || { diff=$((diff+1)); echo "  CHANGED: $f"; }; done < $W/lst
echo "one-variable check: identical=$same changed=$diff (expect changed=1, assets/baseline.joblib)"
sha256sum "$Z"; ls -l "$Z" | awk '{print $5" bytes"}'
rm -rf "$W"
