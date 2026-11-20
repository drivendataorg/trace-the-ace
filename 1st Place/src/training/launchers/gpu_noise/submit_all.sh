#!/usr/bin/env bash
# Submit the GPU-noise study (reports/gpu_noise/report.md) to an IBM LSF cluster, from any directory:
#   bash src/training/launchers/gpu_noise/submit_all.sh
# Results land in reports/gpu_noise/ (one .csv + .json per run; summarise them with
# src/tools/gpu_noise_report.py), job logs in reports/gpu_noise/logs/.
# Set PY to the Python 3.12 environment of README Setup step 2 if `python` is not that one.
# LLMKT_BASE_DIR (another copy of Qwen3-8B) and OUT (another results folder; job logs then go to
# $OUT/logs) are passed on to the jobs when set.
#
# Site-specific: the queue names (p1, gpua100, gpul40s, ...) and the select[] resources
# (gpu80gb, gpu40gb) are those of the DTU HPC cluster we ran on; replace them with your
# cluster's. Our own runs used an equivalent launcher with hard-coded paths and a block D that
# also runs src/run_inference.py end to end on the full test set; this copy is parametrised and
# has blocks A-C only.
#
# Three blocks, so the two sources of difference can be told apart:
#   A  full test set (10,508 rows) at the shipped batch of 8 (2 on cards < 40 GB):
#      what a user of each GPU actually gets
#   B  the same ~1,500-row sample at batch 2 on every GPU: identical batches, so any
#      difference is the hardware
#   C  that sample at batch 1, 4 and 8 on one A100: the effect of the batch alone
#
# Two LSF behaviours this launcher works around: variables exported to
# bsub did not reach the jobs, so SOL/PY/TAG/ROWS/BS are written into each job script after its
# `set -uo pipefail` line (so are LLMKT_BASE_DIR and OUT when set); and a command-line -R
# replaces the script's #BSUB -R lines, so every submission restates rusage/span next to select[].
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
export SOL=$(cd "$HERE/../../../.." && pwd)      # the repository root
export PY=${PY:-python}
LOGS=${OUT:-$SOL/reports/gpu_noise}/logs; mkdir -p "$LOGS"
PASS=""                                          # optional variables, quoted as SOL and PY are
[ -n "${LLMKT_BASE_DIR:-}" ] && PASS="$PASS LLMKT_BASE_DIR='$LLMKT_BASE_DIR'"
[ -n "${OUT:-}" ] && PASS="$PASS OUT='$OUT'"
sub () {  # tag queue select rows batch(empty = automatic)
  local tag=$1 q=$2 sel=$3 rows=$4 bs=${5:-}
  local res="rusage[mem=8GB] span[hosts=1]"; [ -n "$sel" ] && res="select[$sel] $res"
  echo "== $tag ($q ${sel:-any}, rows=$rows, batch=${bs:-auto})"
  sed "/^set -uo pipefail/a export SOL='$SOL' PY='$PY' TAG=$tag ROWS=$rows ${bs:+BS=$bs}$PASS" "$HERE/score.bsub" |
    bsub -q "$q" -J "gpunoise_$tag" -o "$LOGS/$tag.%J.out" -e "$LOGS/$tag.%J.out" -R "$res"
}
# A  full test set, shipped batch
sub full_h100pcie_p1     p1      ""       0    # same GPU model as the reference run: repeat control
sub full_a100_80gb       gpua100 gpu80gb  0
sub full_a100_40gb       gpua100 gpu40gb  0
sub full_l40s            gpul40s ""       0
sub full_a40             gpua40  ""       0
sub full_a10             gpua10  ""       0    # 24 GB: batch 2
# B  ~1,500-row sample, batch 2, every GPU
sub s1500_bs2_h100pcie_p1  p1      ""       1500 2
sub s1500_bs2_a100_80gb    gpua100 gpu80gb  1500 2
sub s1500_bs2_a100_40gb    gpua100 gpu40gb  1500 2
sub s1500_bs2_l40s         gpul40s ""       1500 2
sub s1500_bs2_a40          gpua40  ""       1500 2
sub s1500_bs2_a10          gpua10  ""       1500 2
# V100 (Volta, sm 7.0) is not run: torch 2.11.0+cu128 ships no sm_70 kernels (README, Hardware)
# C  batch effect on one GPU
sub s1500_bs1_a100_80gb    gpua100 gpu80gb  1500 1
sub s1500_bs4_a100_80gb    gpua100 gpu80gb  1500 4
sub s1500_bs8_a100_80gb    gpua100 gpu80gb  1500 8
