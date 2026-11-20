#!/usr/bin/env python
"""Single entry point for the Trace the Ace training pipeline (team appelswim).

Every step that produced a file inside the shipped ``submission.zip`` is listed here, in the
order it ran, with the exact command, the environment it needs, the files it writes, and what
a re-run is expected to reproduce (bit-identical bytes, a functional match, or nothing exact
because the step decodes on a GPU).

It runs inside the pipeline's working directory ``build/workspace/``, which
``src/make_workspace.py`` assembles from the repository (code, shipped intermediate files, and the
raw competition data in ``data/raw/``). Start it through the entry point, which does that first
(``--list`` and ``--help`` are answered without assembling anything):

    python src/run_training.py --list                      # show every step
    python src/run_training.py --check                     # environment + data check
    python src/run_training.py --verify                    # sha-check every intermediate file
    python src/run_training.py --from 03 --to 09           # re-run all CPU steps from distillation on
    python src/run_training.py --steps 05a,07a             # re-run selected steps
    python src/run_training.py --from 02b --to 09          # --from/--to take a stage (02) or a step (02b)
    python src/run_training.py --steps 02a --with-gpu      # one GPU step (README, Run training)

GPU steps (LLM annotation with vLLM, LoRA fine-tuning) are skipped unless ``--with-gpu`` is
given: their outputs ship in the repository (data/interim/, models/), and re-running them costs ~275 GPU-hours
(~240 H100 + ~36 B200) and is not bit-reproducible (vLLM batching / bf16 kernels). Skipped steps are still checked:
their outputs must be present. ``--with-gpu`` does not run the whole chain unattended: two
post-processing scripts were not preserved, so the run stops after 02a (merge the annotation
shards) and after 06c (pool the out-of-fold predictions) and prints
what to do and how to continue. The exit status is 3 when such a stop leaves selected steps
unrun, 0 otherwise.

Before a step overwrites a shipped file, the shipped copy is moved to
``artifacts/_bundle/<same relative path>`` once, so the re-run can be compared against it.
After each step the outputs are compared (sha256 first; for tables and boosters, a functional
comparison when the bytes differ) and a line per output is printed. ``--verify`` checks the
shipped copies (in place, or in ``artifacts/_bundle/`` once a step has moved them) and lists
the files a step rebuilt separately (REBUILT MATCH / REBUILT DIFFERS); it exits non-zero only
for MISSING or CORRUPT files.

Environment variables (each interpreter variable must be a single path to a python executable;
it is quoted when put into a command):
    PY            python of the CPU env (default: the interpreter running this script)
    VLLM_PYTHON   python of the vLLM env used for annotation steps (02*)
    KT_PYTHON     python of the LoRA stage-1 env (requirements-lora-stage1.txt, Python 3.10):
                  steps 01a, 02f (pickle), 06a, the fold parents of 06c, and 07a (for byte
                  equality). Without it, 01a writes the pickle with the CPU env's pandas 3,
                  which the stage-1 env (pandas 1.5.3) reads only thanks to the cast in
                  build_traceace_kt.py, and the bytes differ from the original.
    KT2_PYTHON    python of the LoRA stage-2 env (requirements-lora-stage2.txt, Python 3.12):
                  step 06b and the MRBench continuations + scoring of 06c
    HF_HOME       Hugging Face cache holding the base models (an absolute path). The annotation
                  steps (02*) read it like any Hugging Face program (default
                  ~/.cache/huggingface); the LoRA launchers (06*) default to
                  external/dialogue-kt/.hf. HF_HUB_CACHE, if set, overrides both.
Every GPU step runs offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1: this script sets them
for 02*, run_kt_lsf.sh for 06*) and names its base model by a bare id such as Qwen/Qwen3-8B.
Offline, that id is resolved through ``refs/main`` in the cache, which a download pinned to a
revision hash does not write; download each model at the revision in BASE_MODELS and
write refs/main (README, Setup). The 02* steps also get VLLM_WORKER_MULTIPROC_METHOD=spawn unless it is already
set.
An interpreter variable (PY, VLLM_PYTHON, KT_PYTHON, KT2_PYTHON) that is set must name an
executable; otherwise --check reports an ERROR and no step runs (a dry run only warns).
With ``--with-gpu``, VLLM_PYTHON, KT_PYTHON and KT2_PYTHON must be set for the GPU steps
selected (KT_PYTHON also for 01a), each must have its packages (vllm, plus pyarrow for 02d;
torch, transformers, peft, pykt, pyBKT), and each base model must be cached at its pinned
revision with refs/main naming it; this is checked before any step runs. README, Setup and the
headers of requirements-*.txt set them up.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # the workspace: build/workspace/ (src/make_workspace.py)
REPO = Path(os.environ.get("TTA_REPO") or ROOT.parents[1])   # the repository root (models/, src/, data/)
DKT = "external/dialogue-kt"
SM = f"{DKT}/saved_models"
BUNDLE_BACKUP = ROOT / "artifacts" / "_bundle"
MANIFEST = ROOT / "MANIFEST_intermediates.tsv"

sys.path.insert(0, str(ROOT / "src"))           # unpickling our models needs `magnificat`

PY = os.environ.get("PY", sys.executable)
VLLM_PY = os.environ.get("VLLM_PYTHON", "python")
KT_PY = os.environ.get("KT_PYTHON", PY)
KT2_PY = os.environ.get("KT2_PYTHON", KT_PY)
#: the same, quoted for the bash command strings below (a path with a space stays one word)
qPY, qVLLM, qKT, qKT2 = (shlex.quote(x) for x in (PY, VLLM_PY, KT_PY, KT2_PY))
qREPO, qROOT = shlex.quote(str(REPO)), shlex.quote(str(ROOT))


@dataclass
class Step:
    """One pipeline step: its commands, the files it writes and what a re-run should reproduce."""
    id: str
    title: str
    kind: str                      # "cpu" | "gpu"
    cmds: list[str]                # run with cwd=ROOT through bash
    outputs: list[str]             # paths relative to ROOT
    expect: str                    # "exact" | "functional" | "gpu" | "report"
    est: str                       # wall clock on the hardware we used
    env: dict = field(default_factory=dict)
    note: str = ""
    requires: list[str] = field(default_factory=list)   # inputs that may legitimately be absent
    provided: list[str] = field(default_factory=list)   # what the repository ships for a skipped GPU step
    backup: list[str] = field(default_factory=list)     # further shipped files the step replaces (moved to
                                                         # artifacts/_bundle/ first, not reported as outputs)


V2 = "artifacts/v2labels/122b"
#: Distillation must run with single-threaded BLAS to reproduce the shipped classifiers bit for bit:
#: a multi-threaded BLAS sums in a different order, and the logistic regressions of some
#: cross-fitting folds then stop at different solutions.
ONE = "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
#: The annotation steps (02*) load their model from the local Hugging Face cache only, as the
#: LoRA launcher run_kt_lsf.sh does for 06*. Without these, an annotator would download whatever
#: revision is current upstream instead of the one we ran.
ANNO_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
#: Base model of each GPU step, at the revision we ran: the snapshot paths printed in our job logs
#: (8B: the dialogue-kt cache snapshot printed by the scoring job,
#: LSF job 29267305, as the LoRA logs print only the bare id; 32B:
#: 02b/02c/02f; 122B: 02a, and assumed for 02e, whose log was not kept; 14B: the snapshot in our
#: 02d log). The steps run offline and resolve the bare id through refs/main, so
#: gpu_preflight() checks that refs/main exists and names this revision.
BASE_MODELS = {
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Qwen/Qwen3.5-122B-A10B-FP8": "a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9",
    "Qwen/Qwen2.5-14B-Instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
}
STEP_MODELS = {"02a": "Qwen/Qwen3.5-122B-A10B-FP8", "02b": "Qwen/Qwen3-32B",
               "02c": "Qwen/Qwen3-32B", "02d": "Qwen/Qwen2.5-14B-Instruct",
               "02e": "Qwen/Qwen3.5-122B-A10B-FP8", "02f": "Qwen/Qwen3-32B",
               "06a": "Qwen/Qwen3-8B", "06b": "Qwen/Qwen3-8B", "06c": "Qwen/Qwen3-8B"}
STEPS: list[Step] = [
    # ------------------------------------------------------------------ 01 data preparation
    Step("01a", "Competition data -> dialogue-kt pickle (one row per response, full transcript)",
         "cpu",
         [f"cd {DKT} && {qKT} build_traceace_kt.py --select full --max_turns 0 "
          f"--out data/annotated/traceace_full.pkl"],
         [f"{DKT}/data/annotated/traceace_full.pkl"], "functional", "~5 min",
         note="35,072 responses merged, 35,062 written (the 10 responses of the 5 sessions with no "
              "student turn are skipped). The pickle bytes depend on the pandas version that writes "
              "it (ours: pandas 1.5.3, KT_PYTHON); with --with-gpu, KT_PYTHON is required here so "
              "that 06a/06c read a pickle written by their own env."),
    Step("01b", "Paired-turn text table + regex contingency/repair features",
         "cpu", [f"{qPY} scripts/build_evidence_features.py"],
         ["artifacts/turn_text.pkl.gz", "artifacts/evidence_features.csv"], "exact", "~25 min",
         note="turn_text.pkl.gz (2,368,779 student-anchored turns, truncation 220/220/160) is the "
              "input of every distiller."),
    Step("01c", "Frozen LO-disjoint folds and clean-row mask (sha 59f0c807dad3ac08 / cc365b58eaea4bdb)",
         "cpu",
         [f"{qPY} -c \"import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd; "
          "from magnificat import folds, data; "
          "f=np.load('artifacts/folds_gkf5_lo.npy'); m=np.load('artifacts/clean_rows_gkf5.npy'); "
          "print('folds', folds.fold_sha(f), 'expect', folds.INCUMBENT_SHA); "
          "print('clean', folds.clean_sha(m), 'expect', folds.CLEAN_SHA); "
          "assert folds.fold_sha(f)==folds.INCUMBENT_SHA and folds.clean_sha(m)==folds.CLEAN_SHA\""],
         [], "report", "seconds",
         note="Shipped frozen, not rebuilt: sklearn GroupKFold breaks the 256 tied objective counts "
              "with numpy's unstable argsort, so the partition differs between AVX-512 and older "
              "CPUs. Every training script verifies these shas and refuses to run on a mismatch."),
    # The curriculum table (nc_curriculum.csv) is a gov.uk web scrape, shipped frozen and copied by
    # make_workspace.py; no step builds it.

    # ------------------------------------------------------------------ 02 LLM annotation (GPU)
    Step("02a", "Per-turn answer correctness: Qwen3.5-122B-A10B-FP8, full corpus",
         "gpu",
         [f"cd {DKT} || exit 1; pids=; for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) "
          "ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 ANNO_MAXTOK=4096 ANNO_CHUNK_TURNS=10 "
          f"ANNO_REF=results/NONEXISTENT.csv ANNO_SHARD=$i/4 {qVLLM} annotate_traceace_vllm.py "
          "> results/anno_prod_shard$i.log 2>&1 & pids=\"$pids $!\"; done; rc=0; "
          "for p in $pids; do wait \"$p\" || rc=1; done; "
          "[ $rc = 0 ] || echo 'a shard failed: see results/anno_prod_shard*.log' >&2; exit $rc"],
         [f"{DKT}/results/traceace_turn_labels_vllm_shard{k}.csv" for k in (0, 1, 2, 3)], "gpu",
         "~3 h 35 min on 8xB200", env=ANNO_ENV,
         provided=[f"{DKT}/results/traceace_turn_labels_qwen35_122b.csv"],
         backup=[f"{DKT}/results/traceace_turn_labels_qwen35_122b.csv"],
         note="Writes results/traceace_turn_labels_vllm_shard{0..3}.csv. These were concatenated, "
              "de-duplicated on (session_id, turn) and the chunks whose reply failed to parse "
              "re-run; that merge script was not preserved, so the merged file (2,038,332 rows) "
              "ships frozen. With --with-gpu the run stops after this step so that you merge the "
              "shards yourself."),
    Step("02b", "NTO tutor-move codes: Qwen3-32B, full corpus, 3 shards",
         "gpu",
         [f"cd {DKT} && for s in 0 1 2; do ANNO_SHARD=$s/3 {qVLLM} annotate_nto_moves.py || exit 1; done"],
         [f"{DKT}/results/nto_moves_shard{k}.csv" for k in (0, 1, 2)], "gpu",
         "~18 h per shard on 1xH100", env=ANNO_ENV,
         note="The ntodst_ columns derived from these are dropped from the shipped booster; the "
              "distilled classifier still runs at inference (main.py), so the chain is kept."),
    Step("02c", "Student-state codes: Qwen3-32B, 4,800-session sample (3 x 1,600)",
         "gpu",
         [f"cd {DKT} && for s in 0 1 2; do ANNO_SHARD=$s/3 ANNO_N_SESSIONS=1600 "
          f"{qVLLM} annotate_student_states.py || exit 1; done"],
         [f"{DKT}/results/student_states_shard{k}.csv" for k in (0, 1, 2)], "gpu",
         "~2.7 h per shard on 1xH100", env=ANNO_ENV),
    Step("02d", "Interview-codebook codes: Qwen2.5-14B-Instruct, 4,000 sessions",
         "gpu",
         [f"PYTHONPATH=src {qVLLM} scripts/annotate_doc29_labels.py --qwen Qwen/Qwen2.5-14B-Instruct "
          "--transcripts data/train_transcripts --out-dir artifacts/doc29_records_14b --sessions 4000"],
         ["artifacts/doc29_records_14b/doc29_raw_records_14b.parquet"], "gpu", "~75 min on 1xB200",
         env=ANNO_ENV),
    Step("02e", "Taxonomy-v2 tutor + student moves: Qwen3.5-122B-A10B-FP8, 3,072 sessions",
         "gpu",
         [f"ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 ANNO_ARM=ext ANNO_ROLES=tutor,student "
          f"ANNO_SHARD=0/1 ANNO_N_SESSIONS=3072 ANNO_OUTDIR={V2} {qVLLM} scripts/annotate_v2_moves.py"],
         [f"{V2}/moves_v2_ext_tutor_shard0.csv", f"{V2}/moves_v2_ext_student_shard0.csv"], "gpu",
         "~2-2.5 h on 2xB200", env=ANNO_ENV,
         note="Command reconstructed from the script's settings and the output's shape (the "
              "launcher was not kept). The session list is artifacts/v2_sessions_3072.txt."),
    Step("02f", "MRBench correctness pseudo-labels (Qwen3-32B) -> stage-2 LoRA training pickle",
         "gpu",
         [f"cd {DKT} && {qVLLM} annotate_mrbench_vllm.py && {qKT} build_mrbench_pkl.py"],
         [f"{DKT}/results/mrbench_turn_labels_qwen32b.csv", f"{DKT}/data/annotated/mrbench_train.pkl",
          f"{DKT}/data/annotated/mrbench_test.pkl"], "gpu", "~13 min on 1xH100", env=ANNO_ENV,
         note="Reads the MRBench JSON files (CC BY-SA 4.0) that src/make_workspace.py copies from "
              "data/external/mrbench/raw/ into external/datasets/mrbench/."),

    # ------------------------------------------------------------------ 03 distillation (CPU)
    Step("03a", "Distil the correctness annotator -> turn_correctness_clf + recdst_ block",
         "cpu", [f"{ONE} {qPY} scripts/distil_turn_correctness.py --overwrite-shipped"],
         ["artifacts/turn_correctness_clf.joblib", "artifacts/turn_correctness_probs.pkl.gz",
          "artifacts/evidence_features_full.csv"], "functional", "~26-40 min, 1 core",
         note="The classifier files of 03a, 03c, 03d and 03e are byte-identical to the shipped ones only "
              "under the same CPython build: each fitted vectorizer stores _stop_words_id = id(None), a "
              "constant of the interpreter binary (0x1443f80 in the python-build-standalone build "
              "20260623 of CPython 3.12.13 that wrote the shipped files; print yours with python -c "
              "'print(hex(id(None)))'). With another build they differ in that field only and score "
              "every turn identically (the per-turn tables below are EQUAL). That holds where numpy's and "
              "scipy's OpenBLAS runs its SkylakeX kernels (the AVX-512 CPUs we used); with its Haswell "
              "kernels (AVX2-only CPUs) 03a-03f fit different weights and everything downstream "
              "differs (README, \"Byte-identical or value-identical\")."),
    Step("03b", "Session-level oracle NTO rates (diagnostic input of 03c)",
         "cpu", [f"{qPY} scripts/build_nto_rate_features.py"],
         ["artifacts/nto_rate_features.csv"], "exact", "~2 min"),
    Step("03c", "Distil the NTO annotator -> nto_moves_clf + ntodst_ block",
         "cpu", [f"{ONE} {qPY} scripts/distil_nto_moves.py --overwrite-shipped"],
         ["artifacts/nto_moves_clf.joblib", "artifacts/nto_turn_probs.pkl.gz",
          "artifacts/nto_dst_features.csv"], "functional", "~25-60 min, 1 core"),
    Step("03d", "Distil the student-state annotator -> stu_states_clf + studst_ block",
         "cpu", [f"{ONE} {qPY} scripts/distil_student_states.py --overwrite-shipped"],
         ["artifacts/stu_states_clf.joblib", "artifacts/stu_turn_probs.pkl.gz",
          "artifacts/stu_dst_features.csv"], "functional", "~21-60 min, 1 core",
         note="Reads 03a/03c per-turn tables for side columns, so it runs after them."),
    Step("03e", "Distil the interview-codebook annotator -> doc29_clf",
         "cpu", [f"{ONE} {qPY} scripts/distil_doc29_labels.py"],
         ["artifacts/doc29_clf.joblib", "artifacts/doc29_turn_probs.pkl.gz"], "functional",
         "~35 min"),
    Step("03f", "Distil the taxonomy-v2 annotator -> v2t_clf (tutor) + v2s_clf (student)",
         "cpu",
         [f"{ONE} {qPY} scripts/distil_v2_moves.py --labels {V2}/moves_v2_ext_tutor_shard0.csv "
          "--map v2_ext --role tutor --prefix v2t --tag v2ext_122bC "
          "--restrict-sessions artifacts/v2_sessions_3072.txt --ratios v2",
          f"{ONE} {qPY} scripts/distil_v2_moves.py --labels {V2}/moves_v2_ext_student_shard0.csv "
          "--map student --role student --prefix v2s --tag v2stu_122bC "
          "--restrict-sessions artifacts/v2_sessions_3072.txt"],
         ["artifacts/v2t_clf_v2ext_122bC.joblib", "artifacts/v2s_clf_v2stu_122bC.joblib",
          "artifacts/v2t_turn_probs_v2ext_122bC.pkl.gz", "artifacts/v2s_turn_probs_v2stu_122bC.pkl.gz",
          "artifacts/v2t_features_v2ext_122bC.meta.json", "artifacts/v2s_features_v2stu_122bC.meta.json"],
         "functional", "~44-51 min (tutor then student, one after the other; the original cluster run took 31 min with both at once)",
         note="The two classifier files differ from the shipped ones in bytes (also after their "
              "vocabularies are stored sorted), but they score every turn identically: the turn "
              "probability tables are equal and the meta files match (on AVX-512 CPUs; see the "
              "03a note)."),

    # ------------------------------------------------------------------ 04 session feature tables
    Step("04a", "Interview-codebook session block (34 d29dst_ columns)",
         "cpu", [f"OMP_NUM_THREADS=1 {qPY} scripts/build_doc29_features.py --mode distilled"],
         ["artifacts/doc29_features_distilled.csv"], "functional", "~6 min"),
    Step("04b", "Taxonomy-v2 session block, kappa-shrunk (46 v2ht_/v2hs_ columns)",
         "cpu",
         [f"OMP_NUM_THREADS=4 PYTHONPATH=src {qPY} scripts/build_v2_families.py --tag 122bC --jobs 8 "
          "--tutor artifacts/v2t_turn_probs_v2ext_122bC.pkl.gz "
          "--student artifacts/v2s_turn_probs_v2stu_122bC.pkl.gz"],
         ["artifacts/v2fam_shrunk_122bC.csv"], "functional", "~5 min",
         note="--tutor/--student are needed: the script defaults point at the un-suffixed _122b "
              "tables, which build a different block."),
    Step("04c", "Train/serve parity gates for the distilled blocks (must all exit 0)",
         "cpu",
         [f"PYTHONPATH=src {qPY} scripts/check_v2_family_parity.py --families shrunk",
          f"PYTHONPATH=src {qPY} scripts/check_studst_parity.py --n 400",
          f"PYTHONPATH=src {qPY} scripts/check_doc29_parity.py --n 400"],
         [], "report", "~10 min",
         note="Each gate re-scores sessions through the inference code path (the module main.py "
              "uses) and compares the result with the training table; offline CV cannot detect "
              "an inference copy that has drifted. "
              "The gates cover the d29dst_ (34), studst_ (12) and v2ht_/v2hs_ (46) blocks: 92 of "
              "the 177 booster columns. The other 85 (54 handcrafted, 14 recdst_, 10 ctg_, 7 rep_) "
              "have no numeric gate; recdst_, ctg_ and rep_ are built for training by separate "
              "code (scripts/distil_turn_correctness.py, scripts/build_evidence_features.py). "
              "src/tools/check_train_serve_equivalence.py compares the two copies of "
              "the feature modules."),

    # ------------------------------------------------------------------ 05 XGBoost half
    Step("05a", "Train the XGBoost half (177 columns, 5 seeds, frozen Platt)",
         "cpu",
         [f"OMP_NUM_THREADS=1 {qPY} scripts/train_model_e3l.py --n-seeds 5 --n-jobs 12 --par 4 "
          "--no-loknn --swap-student --extra-no-ablate --drop-cols ntodst_ "
          "--extra-csv d29dst_=artifacts/doc29_features_distilled.csv "
          "--extra-csv v2h=artifacts/v2fam_shrunk_122bC.csv "
          "--out artifacts/baseline_v2shrunkD.joblib "
          "--oof-out artifacts/v2shrunkD_oof.csv"],
         ["artifacts/baseline_v2shrunkD.joblib", "artifacts/v2shrunkD_oof.csv"], "functional",
         "~44 min, 12 cores, 10 GB",
         note="Deterministic: the CPU rebuilds of 2026-09-02 and 2026-10-17 produced byte-identical "
              "boosters (5 seeds) and OOF; the joblib file itself differs from the shipped one in "
              "bytes only and loads to the same model, given the same 03/04 inputs (on an AVX2-only "
              "CPU those differ; see the 03a note)."),
    Step("05b", "Score the XGBoost half on LO-disjoint pooled and clean rows",
         "cpu",
         [f"OMP_NUM_THREADS=4 {qPY} scripts/rescore_clean.py "
          "--cand v2shrunkD=artifacts/v2shrunkD_oof.csv --tag _v2shrunkD"],
         [], "report", "~3 min",
         note="Expect Platt LL 0.5718 / AUROC 66.61 pooled; 0.6295 / 64.50 on the 16,687 XGBoost-clean "
              "rows."),

    # ------------------------------------------------------------------ 06 LLM half (GPU)
    Step("06a", "LoRA stage 1: Qwen3-8B on all 35,062 competition dialogues, 2 epochs",
         "gpu",
         [f"cd {DKT} && TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 "
          f"KT_PYTHON={qKT} ./run_kt_lsf.sh train --dataset traceace --model_type lmkt --train_all "
          "--pack_kcs 0 --quantize 0 --inc_first_label --epochs 2 --base_model Qwen/Qwen3-8B "
          "--model_name lmkt_traceace_qwen3_8b_full_final"],
         [f"{SM}/lmkt_traceace_qwen3_8b_full_final/{n}"
          for n in ("adapter_config.json", "adapter_model.safetensors")], "gpu",
         "51.5 h on 1xH100-80GB"),
    Step("06b", "LoRA stage 2: 1 epoch on MRBench (17 optimizer steps) -> the shipped adapter",
         "gpu",
         [f"cd {DKT} && TRACEACE_PKL=data/annotated/mrbench_train.pkl LLMKT_MAX_SEQ_LEN=8192 "
          f"KT_PYTHON={qKT2} ./run_kt_lsf.sh train --dataset traceace --model_type lmkt --train_all "
          "--pt_model_name lmkt_traceace_qwen3_8b_full_final --pack_kcs 0 --quantize 0 "
          "--inc_first_label --epochs 1 --lr 1e-4 --base_model Qwen/Qwen3-8B "
          "--model_name lmkt_mrft_sa_8b_all35k"],
         [f"{SM}/lmkt_mrft_sa_8b_all35k/{n}"
          for n in ("adapter_config.json", "adapter_model.safetensors", "train_log.csv")], "gpu",
         "~15 min on 1xB200"),
    Step("06c", "OOF twins for the calibrator: 3 LO-disjoint fold parents + MRBench + held-out scoring",
         "gpu",
         [f"cd {DKT} && for k in 1 2 3; do "
          f"TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON={qKT} "
          "./run_kt_lsf.sh train --dataset traceace --model_type lmkt --fold $k --pack_kcs 0 "
          "--quantize 0 --inc_first_label --epochs 2 --base_model Qwen/Qwen3-8B --model_name lmkt_traceace_qwen3_8b_full && "
          f"TRACEACE_PKL=data/annotated/mrbench_train.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON={qKT2} ./run_kt_lsf.sh train "
          "--dataset traceace --model_type lmkt --train_all --pt_model_name lmkt_traceace_qwen3_8b_full_$k "
          "--pack_kcs 0 --quantize 0 --inc_first_label --epochs 1 --lr 1e-4 --base_model Qwen/Qwen3-8B "
          "--model_name lmkt_mrft_sa_8b_a1f$k && "
          "ln -sfn lmkt_mrft_sa_8b_a1f$k saved_models/lmkt_mrft_sa_8b_a1f${k}_$k && "
          f"TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON={qKT2} ./run_kt_lsf.sh test "
          "--dataset traceace --model_type lmkt --fold $k --batch_size 8 --pack_kcs 0 --quantize 0 "
          "--inc_first_label --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f$k || exit 1; done"],
         [f"{DKT}/results/qual_lmkt_mrft_sa_8b_a1f{k}_incfirst_{k}.csv" for k in (1, 2, 3)]
         + [f"{SM}/lmkt_traceace_qwen3_8b_full_{k}/{n}" for k in (1, 2, 3)
            for n in ("adapter_config.json", "adapter_model.safetensors")]
         + [f"{SM}/lmkt_mrft_sa_8b_a1f1/{n}"
            for n in ("adapter_config.json", "adapter_model.safetensors", "train_log.csv")], "gpu",
         "~41 h per fold parent on 1xH100 (3 in parallel); MRBench continuation + scoring ~10 min + "
         "~1 h 31 for fold 1 on 1xH100, ~15 + ~35 min per fold for folds 2-3 on 1xB200",
         provided=["artifacts/mrft_oof_3fold.csv", "artifacts/mrft_oof_3fold_keyed.csv"],
         backup=["artifacts/mrft_oof_3fold.csv"],
         note="Pooled into artifacts/mrft_oof_3fold.csv (fold,y,p; n=16,693). Neither the pooling "
              "script nor the original fold-2/3 qual files were preserved, so the pooled "
              "CSV ships frozen and is the calibrator's sole input. With --with-gpu the run stops "
              "after this step so that you pool the new qual files yourself. The original "
              "fold-1 MRBench continuation and scoring ran in the stage-1 env; this "
              "command uses KT2_PYTHON for all three folds."),

    # ------------------------------------------------------------------ 07 calibration
    Step("07a", "Frozen Platt map for the LLM half (a = min(OOF fit, 0.85); b anchors train base rate)",
         "cpu",
         [f"{ONE} {qKT} scripts/fit_calibrator.py --model mrft --oof-csv artifacts/mrft_oof_3fold.csv "
          "--out artifacts/llmkt_calibrator_mrft_sub0808.joblib"],
         ["artifacts/llmkt_calibrator_mrft_sub0808.joblib"], "exact", "seconds",
         note="Single-threaded BLAS. Byte-identical to the shipped file (a=0.6876178756942771 "
              "b=0.45650491437726304) when fit in the dialogue-kt env (KT_PYTHON: py3.10, numpy 1.26.4) "
              "on an AVX-512 Intel CPU with 1-6 BLAS threads (measured 2026-10-21 on Xeon Gold 6126, "
              "6142, 6226R). The last bits depend on the numpy build, the CPU and the BLAS thread count: "
              "with 8+ threads or on an AVX2-only CPU a and b land 1 and 2 units in the last place away "
              "(…772 / …315), and in the py3.12 CPU env up to 7 away (b …634 on AVX-512). Every "
              "prediction then moves by ~1e-16; the step prints DIFFERS with both values."),
    Step("07b", "Key the pooled LLM OOF by response_id (needs the 06c qual files)",
         "cpu", [f"{qPY} scripts/build_mrft_oof.py"],
         ["artifacts/mrft_oof_3fold_keyed.csv"], "functional", "seconds",
         note="Only meaningful after re-running 06c; otherwise the shipped keyed CSV is used.",
         requires=[f"{DKT}/results/qual_lmkt_mrft_sa_8b_a1f{k}_incfirst_{k}.csv" for k in (1, 2, 3)]),

    # ------------------------------------------------------------------ 08 blend weight (report)
    Step("08a", "Blend report: LL/AUROC of LLM, XGBoost, and p = 0.57 p_llm + 0.43 p_gbm on clean rows",
         "cpu",
         [f"OMP_NUM_THREADS=4 {qPY} scripts/e3_blend_decide.py --clean-only "
          "--llm-calib llmkt_calibrator_mrft_sub0808.joblib "
          "--cand v2shrunkD=artifacts/v2shrunkD_oof.csv "
          "--tag _repro"],
         [], "report", "~5 min",
         note="w = 0.43 is fixed (models/submission/assets/blend_weight.txt); "
              "this step writes a report (figures/data/e3_blend_decide_repro.md), it does not refit "
              "the weight."),

    # ------------------------------------------------------------------ 09 assemble the submission
    Step("09a", "Stage trained artifacts into a submission tree under their shipped names",
         "cpu", [f"{qPY} {qREPO}/src/tools/stage_assets.py --training-root {qROOT} --out {qROOT}/build/submission"],
         [], "report", "seconds",
         note="Copies the artifacts into build/submission/assets/ and reports, per file, whether it "
              "is byte-identical to the shipped assets/ (which retrained files can differ in bytes, and "
              "why: the 09b note). It fails when a trained file is missing "
              "from the workspace and only the shipped copy in artifacts/_bundle/ is left "
              "(printed as FALLBACK; stage_assets.py --allow-fallback accepts that)."),
    Step("09b", "Pack submission.zip (sorted sha256 manifest, zip -X) and compare with the shipped tree",
         "cpu", [f"COMPARE_REPORT_ONLY=1 bash {qREPO}/src/tools/package_submission.sh {qROOT}/build/submission {qROOT}/build/submission.zip"],
         [], "report", "~1 min",
         note="Lists every zip entry that differs from models/submission.zip without failing. After a "
              "quick check all 49 file entries are identical when 07a printed MATCH; otherwise the "
              "calibrator and therefore assets/manifest_sha256.txt differ (47/49; see the 07a note). "
              "After the full CPU rebuild the XGBoost model and the two v2 move classifiers also differ "
              "in bytes (identical boosters and turn probabilities, see 05a/03f), and so do the four "
              "classifiers of 03a-03e unless the interpreter is the CPython build that wrote them (see "
              "the 03a note): 45/49 at best, 40/49 with another build and a differing calibrator "
              "(release check 2026-10-21). On an AVX2-only CPU the same 40/49, but the classifiers and "
              "the XGBoost model then differ in value too (release check 2026-10-27; see the 03a note)."),
]
BY_ID = {s.id: s for s in STEPS}


# ---------------------------------------------------------------------------------- helpers
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_manifest() -> dict[str, tuple[int, str]]:
    """Read ``MANIFEST_intermediates.tsv`` from the workspace.

    Returns:
        ``{workspace path: (size in bytes, sha256)}``; empty if the manifest is absent.
    """
    if not MANIFEST.exists():
        return {}
    out = {}
    for ln in MANIFEST.read_text().splitlines()[1:]:
        path, size, digest, *_ = ln.split("\t")
        out[path] = (int(size), digest)
    return out


def functional_compare(new: Path, old: Path) -> str:
    """Compare a rebuilt output with the shipped copy when their bytes differ.

    CSV tables and pickled frames are compared by value (``EQUAL`` below 1e-9, ``CLOSE`` below
    1e-6), the XGBoost model by the JSON dump of each seed's booster, a Platt calibrator by its
    two floats.

    Args:
        new: the file the step wrote.
        old: the shipped copy (in ``artifacts/_bundle/``).

    Returns:
        A one-line verdict for the step report, e.g. ``"EQUAL (max|d|=0.00e+00)"``.
    """
    try:
        import pandas as pd
        if new.suffix == ".csv":
            a, b = pd.read_csv(new), pd.read_csv(old)
            if list(a.columns) != list(b.columns) or len(a) != len(b):
                return f"DIFFERENT shape/columns {a.shape} vs {b.shape}"
            num = a.select_dtypes("number").columns
            d = (a[num].astype(float) - b[num].astype(float)).abs().max().max() if len(num) else 0.0
            same_obj = a.drop(columns=num).equals(b.drop(columns=num))
            return f"{'EQUAL' if d < 1e-9 and same_obj else 'CLOSE' if d < 1e-6 else 'DIFFERENT'} (max|d|={d:.2e})"
        if new.name.endswith(".pkl.gz") or new.suffix == ".pkl":
            comp = "gzip" if new.name.endswith(".gz") else None
            a, b = pd.read_pickle(new, compression=comp), pd.read_pickle(old, compression=comp)
            if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
                if list(a.columns) != list(b.columns) or len(a) != len(b):
                    return f"DIFFERENT shape/columns {a.shape} vs {b.shape}"
                num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
                txt = [c for c in a.columns if c not in num]
                # string columns are compared by value: pandas may store them as python or
                # pyarrow strings depending on what is installed, which changes the pickle only
                same_txt = all((a[c].astype(object).fillna("") == b[c].astype(object).fillna("")).all()
                               for c in txt)
                d = float((a[num].astype(float) - b[num].astype(float)).abs().max().max()) if num else 0.0
                tag = "EQUAL" if same_txt and d < 1e-9 else "CLOSE" if same_txt and d < 1e-6 else "DIFFERENT"
                return f"{tag} (values; max|d|={d:.2e}{'' if same_txt else ', text differs'})"
            if hasattr(a, "equals"):
                return "EQUAL (values)" if a.equals(b) else "DIFFERENT (values)"
        if new.suffix == ".joblib":
            import joblib
            a, b = joblib.load(new), joblib.load(old)
            if hasattr(a, "bases") and hasattr(b, "bases"):
                ja = [m.named_steps["clf"].get_booster().save_raw("json") for m in a.bases]
                jb = [m.named_steps["clf"].get_booster().save_raw("json") for m in b.bases]
                same = all(x == y for x, y in zip(ja, jb)) and len(ja) == len(jb)
                return f"{'BOOSTERS IDENTICAL' if same else 'BOOSTERS DIFFER'} ({len(ja)} seeds)"
            if hasattr(a, "a") and hasattr(a, "b"):
                return f"Platt a={a.a!r} b={a.b!r} vs a={b.a!r} b={b.b!r}"
    except Exception as e:  # noqa: BLE001
        return f"compare failed: {type(e).__name__}: {e}"
    return "DIFFERENT bytes"


def backup_bundled(paths: list[str]) -> None:
    """Move each shipped file a step is about to overwrite to ``artifacts/_bundle/``, once.

    Args:
        paths: workspace-relative paths; files that do not exist, or whose backup already
            exists, are left alone.

    Returns:
        None.
    """
    for rel in paths:
        src, dst = ROOT / rel, BUNDLE_BACKUP / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))


def report_outputs(step: Step, manifest) -> int:
    """Print one line per declared output of a step that has just run.

    Each output is reported as ``MATCH`` (sha256 of the shipped file), ``WROTE`` (not in the
    manifest), ``REBUILT`` (a regenerated competition-text file) or ``DIFFERS`` with the result of
    ``functional_compare``.

    Args:
        step: the step that ran.
        manifest: the result of ``load_manifest``.

    Returns:
        The number of declared outputs that are missing.
    """
    n_missing = 0
    for rel in step.outputs:
        p = ROOT / rel
        if not p.exists():
            print(f"    MISSING   {rel}")
            n_missing += 1
            continue
        got = sha256(p)
        want = manifest.get(rel, (None, None))[1]
        if want is None:
            print(f"    WROTE     {rel}  sha256={got[:16]}  (not in manifest)")
        elif got == want:
            print(f"    MATCH     {rel}  (sha256 identical to the shipped file)")
        elif rel in not_shipped():
            print(f"    REBUILT   {rel}  (not shipped: competition text; its bytes differ from the "
                  f"original's sha256, as expected when pandas stores strings differently; "
                  f"01a matches byte for byte under KT_PYTHON)")
        else:
            old = BUNDLE_BACKUP / rel
            how = functional_compare(p, old) if old.exists() else "no shipped copy to compare"
            print(f"    DIFFERS   {rel}  -> {how}")
    return n_missing


#: GPU steps that cannot be followed automatically: the run stops after them (see the docstring)
STOP_AFTER = {
    "02a": ("merge the four shard files results/traceace_turn_labels_vllm_shard{0..3}.csv into "
            "results/traceace_turn_labels_qwen35_122b.csv (in build/workspace/external/dialogue-kt/). "
            "A plain concatenation and de-duplication is approximate: "
            "the original merge also re-ran the ~14 % of chunks whose reply failed to parse. To "
            "keep the shipped merged file instead, copy it back from artifacts/_bundle/"
            f"{DKT}/results/."),
    "06c": ("pool the three new qual files into artifacts/mrft_oof_3fold.csv (columns fold,y,p): "
            "per file, the last row with a numeric Prob for each Dialogue "
            "ID (the extraction of scripts/build_mrft_oof.py terminal_preds), y = Correct as 0/1, "
            "p = Prob, folds 1, 2, 3 in that order. The shipped pooled file is in artifacts/_bundle/"
            "artifacts/. Then continue with: python src/run_training.py --steps 07b,07a,08a,09a,09b"),
}


def stop_message(sid: str, rest: list[str]) -> str:
    """Build the message printed when the run stops after a GPU step.

    Args:
        sid: the step id, a key of ``STOP_AFTER``.
        rest: ids of the selected steps that did not run.

    Returns:
        ``STOP_AFTER[sid]`` plus, for 02a, the command that continues with ``rest``.
    """
    msg = STOP_AFTER[sid]
    if sid != "02a":
        return msg
    if rest:
        gpu = any(BY_ID[r].kind == "gpu" for r in rest)
        return (f"{msg} Then continue with: python src/run_training.py --steps {','.join(rest)}"
                + (" --with-gpu" if gpu else ""))
    return (f"{msg} Then re-run the CPU steps that read it: python src/run_training.py --from 03 "
            "--to 09 (use --from 02b --to 09 --with-gpu to also redo the other GPU steps)")

#: interpreters each step needs with --with-gpu, and the packages each must provide (find_spec)
LORA = ("torch", "transformers", "peft", "pykt", "pyBKT")
VLLM = ("vllm",)
STEP_ENVS = {"01a": {"KT_PYTHON": ("pandas",)},
             **{i: {"VLLM_PYTHON": VLLM} for i in ("02a", "02b", "02c", "02e")},
             "02d": {"VLLM_PYTHON": VLLM + ("pyarrow",)},     # 02d writes parquet
             "02f": {"VLLM_PYTHON": VLLM, "KT_PYTHON": ("pandas",)},
             "06a": {"KT_PYTHON": LORA}, "06b": {"KT2_PYTHON": LORA},
             "06c": {"KT_PYTHON": LORA, "KT2_PYTHON": LORA}}


def _steps(ids: list[str]) -> str:
    return ("steps " if len(ids) > 1 else "step ") + ", ".join(ids)


def hub_cache(step_id: str) -> Path:
    """Return the Hugging Face hub cache a GPU step reads.

    That is ``HF_HUB_CACHE`` when set, else ``$HF_HOME/hub``, where ``HF_HOME`` defaults to
    ``~/.cache/huggingface`` for the annotators (02*) and, as in ``run_kt_lsf.sh``, to
    ``external/dialogue-kt/.hf`` for the LoRA steps (06*).

    Args:
        step_id: the step id, e.g. ``"02a"`` or ``"06b"``.

    Returns:
        The cache folder.
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    home = os.environ.get("HF_HOME")
    if not home:
        home = (ROOT / DKT / ".hf" if step_id.startswith("06") else
                Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    return Path(home).expanduser() / "hub"


def model_cache_problems(steps: list[Step]) -> list[str]:
    """Check that every base model the selected GPU steps load is cached at its pinned revision.

    The GPU steps run offline and load a bare model id, which the Hugging Face libraries resolve
    through ``refs/main`` in the cache. A download pinned to a revision hash writes the snapshot
    but no ``refs/main``, and the step then fails with "couldn't find them in the cached files".
    Per model, ``refs/main`` must exist and name the revision in ``BASE_MODELS``, and the snapshot
    must hold its config and weights.

    Args:
        steps: the selected steps; only those in ``STEP_MODELS`` are checked.

    Returns:
        One message per problem; empty when every model is ready.
    """
    need: dict[tuple[str, Path], list[str]] = {}
    for s in steps:
        if s.id in STEP_MODELS:
            need.setdefault((STEP_MODELS[s.id], hub_cache(s.id)), []).append(s.id)
    if not need:
        return []
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(var)
        if v and not Path(v).expanduser().is_absolute():
            return [f"{var}={v} is a relative path; the steps run from different directories, so "
                    "give an absolute path"]
    problems = []
    for (model, cache), ids in need.items():
        rev = BASE_MODELS[model]
        repo = cache / f"models--{model.replace('/', '--')}"
        ref = repo / "refs" / "main"
        snap = repo / "snapshots" / rev
        # huggingface_hub reads refs/main with a bare read() and no strip, so compare raw text.
        raw = ref.read_text() if ref.is_file() else None
        if not repo.is_dir():
            why = f"it is not in the cache {cache}"
        elif raw is None:
            why = f"{ref} does not exist"
        elif raw != rev and raw.strip() == rev:
            why = (f"{ref} has a trailing newline or space; write it with printf %s '<hash>', "
                   "not echo")
        elif raw != rev:
            why = f"{ref} names {raw.strip()[:12]}, not the pinned revision"
        elif not (snap / "config.json").exists() or not any(snap.glob("*.safetensors")):
            why = f"the snapshot {snap} lacks config.json or the *.safetensors weights"
        else:
            continue
        problems.append(f"{model} @ {rev[:12]} (needed by {_steps(ids)}): {why}. Download it at that "
                        "revision and write the revision into refs/main (README, Setup)")
    return problems


def gpu_preflight(steps: list[Step]) -> list[str]:
    """Check the GPU environments of the selected steps before ``--with-gpu`` runs anything.

    The interpreter variables the steps need (``STEP_ENVS``) must be set explicitly and must
    provide their packages, and the base models must be cached at their pinned revision
    (``model_cache_problems``).

    Args:
        steps: the selected steps.

    Returns:
        One message per problem; empty when all is well.
    """
    need: dict[str, tuple[list[str], set[str]]] = {}
    for s in steps:
        for var, mods in STEP_ENVS.get(s.id, {}).items():
            ids, allm = need.setdefault(var, ([], set()))
            ids.append(s.id)
            allm.update(mods)
    problems = []
    for var, (ids, mods) in need.items():
        exe = os.environ.get(var)
        if not exe:
            problems.append(f"{var} is not set (needed by {_steps(ids)})")
            continue
        code = ("import importlib.util, sys; m = [x for x in sys.argv[1:] "
                "if importlib.util.find_spec(x) is None]; print(' '.join(m)); sys.exit(1 if m else 0)")
        try:
            r = subprocess.run([exe, "-c", code, *sorted(mods)], capture_output=True, text=True,
                               timeout=120)
        except (OSError, subprocess.TimeoutExpired) as e:
            problems.append(f"{var}={exe} cannot be run: {e}")
            continue
        if r.returncode != 0:
            problems.append(f"{var}={exe} lacks {r.stdout.strip() or 'a working python'} "
                            f"(needed by {_steps(ids)})")
    return problems + model_cache_problems(steps)


def interpreter_problems() -> list[str]:
    """Check that each interpreter variable that is set names an executable.

    A path or a command on PATH is accepted. Otherwise the first step that uses it would fail,
    possibly hours into a run.

    Returns:
        One message per bad variable among PY, VLLM_PYTHON, KT_PYTHON and KT2_PYTHON.
    """
    out = []
    for var in ("PY", "VLLM_PYTHON", "KT_PYTHON", "KT2_PYTHON"):
        v = os.environ.get(var)
        if v is not None and not shutil.which(v):
            out.append(f"{var}={v!r} is set but is not an executable file. Unset it, or export it "
                       "only once the interpreter exists, e.g. "
                       "[ -x .venv-kt/bin/python ] && export KT_PYTHON=$PWD/.venv-kt/bin/python")
    return out


def check_environment() -> int:
    """Print the ``--check`` report: Python, pinned CPU packages and the raw training data.

    Also flags a stray ``artifacts/lo_curriculum.json`` and a workspace path with whitespace.

    Returns:
        The number of problems found (version mismatches, missing packages or data files).
    """
    bad = 0
    print(f"python  {sys.version.split()[0]}  ({PY})")
    if sys.version_info[:2] != (3, 12):
        print("  WARN: the CPU pipeline was run on Python 3.12")
    pins = {"numpy": "2.2.6", "pandas": "3.0.3", "sklearn": "1.8.0", "xgboost": "3.3.0",
            "scipy": "1.17.1", "joblib": "1.5.3", "pyarrow": "25.0.0"}
    for mod, want in pins.items():
        try:
            got = __import__(mod).__version__
            flag = "ok" if got == want else "MISMATCH"
            bad += flag != "ok"
            print(f"  {mod:8s} {got:10s} pinned {want:8s} {flag}")
        except ImportError:
            print(f"  {mod:8s} MISSING (pinned {want})"); bad += 1
    data = ROOT / "data"
    for pat in ("train_features*.csv", "train_labels*.csv"):
        hits = list(data.glob(pat))
        print(f"  data/{pat:24s} {'ok' if len(hits) == 1 else 'MISSING/AMBIGUOUS'}")
        bad += len(hits) != 1
    tdir = data / "train_transcripts"
    n = len(list(tdir.glob("*.csv"))) if tdir.is_dir() else 0
    print(f"  data/train_transcripts/      {n} csv files "
          + ("ok" if n >= 22800 else "INCOMPLETE: put the competition's train_transcripts*.zip (or the "
             "unzipped train_transcripts/ folder) in data/raw/; the next run_training.py call links it"))
    bad += n < 22800
    if (ROOT / "artifacts/lo_curriculum.json").exists():
        print("  ERROR: artifacts/lo_curriculum.json exists. It silently switches the curriculum "
              "features to an LLM-annotated tier the shipped model never saw. Delete it.")
        bad += 1
    if " " in str(ROOT) or "\t" in str(ROOT):
        print(f"  WARN: the workspace path contains whitespace ({ROOT}). The commands here quote it, "
              "but some vendored research scripts may not; a path without spaces is safer.")
    return bad


def not_shipped() -> dict[str, str]:
    """List the files the repository does not ship because they are competition transcripts.

    Read from ``workspace_map.tsv`` (rows whose ``how`` is ``regenerate:<step>``).

    Returns:
        ``{workspace path: id of the step that regenerates it from data/raw/}``.
    """
    m = ROOT / "workspace_map.tsv"
    if not m.exists():
        return {}
    out = {}
    for ln in m.read_text().splitlines()[1:]:
        f = ln.split("\t")
        if f[2].startswith("regenerate:"):
            out[f[0]] = f[2].split(":", 1)[1]
    return out


def intact(p: Path, size: int, digest: str) -> bool:
    """Return True if the file has the given size and sha256."""
    return p.stat().st_size == size and sha256(p) == digest


def verify_bundle(manifest) -> int:
    """Check the sha256 of every shipped intermediate file (``--verify``).

    Once a step has moved a shipped file to ``artifacts/_bundle/``, the copy there is the one
    checked, and the file the step wrote in its place is listed separately (``REBUILT MATCH`` /
    ``REBUILT DIFFERS``, never counted as a failure). Files the repository does not ship
    (competition transcripts) are listed as rebuilt or ``NOT YET``.

    Args:
        manifest: the result of ``load_manifest``.

    Returns:
        The number of shipped files that are missing or corrupt.
    """
    regen = not_shipped()
    made_by = {o: s.id for s in STEPS for o in s.outputs}
    moved_by = {o: s.id for s in STEPS for o in s.backup}
    n_ok = n_bad = n_pending = n_same = n_diff = 0

    def rebuilt(rel, p, digest):
        nonlocal n_same, n_diff
        if rel in made_by:
            by = f"written by step {made_by[rel]}"
        elif rel in moved_by:
            by = (f"replaced after step {moved_by[rel]} moved the shipped copy aside (written by hand "
                  "or by a later step)")
        else:
            by = "written by a step"
        if sha256(p) == digest:
            n_same += 1
            print(f"REBUILT MATCH    {rel}  ({by}; identical to the original)")
        else:
            n_diff += 1
            print(f"REBUILT DIFFERS  {rel}  ({by}; see the step's note and README, Run training for "
                  "which re-runs are expected to differ)")

    for rel, (size, digest) in manifest.items():
        live, kept = ROOT / rel, BUNDLE_BACKUP / rel
        if rel in regen:                        # not shipped: only a regenerated copy can exist
            if live.exists():
                rebuilt(rel, live, digest)
            elif kept.exists():
                rebuilt(rel, kept, digest)
            else:
                n_pending += 1
                print(f"NOT YET  {rel}  (competition transcripts, not shipped: run step {regen[rel]})")
            continue
        if kept.exists():                       # a step moved the shipped copy aside
            good = intact(kept, size, digest)
            print(f"{'OK     ' if good else 'CORRUPT'}  {rel}  (shipped copy in artifacts/_bundle/)")
            if not live.exists():
                print(f"MISSING  {rel}  (moved to artifacts/_bundle/ by a step that did not write it "
                      "again: re-run that step, or move the shipped copy back)")
                good = False
            else:
                rebuilt(rel, live, digest)
            n_ok += good
            n_bad += not good
        elif not live.exists():
            n_bad += 1
            print(f"MISSING  {rel}")
        elif intact(live, size, digest):
            n_ok += 1
        else:
            n_bad += 1
            print(f"CORRUPT  {rel}")
    n_shipped = sum(rel not in regen for rel in manifest)
    print(f"{n_ok}/{n_shipped} shipped intermediate files verified"
          + (f"; {n_bad} MISSING or CORRUPT" if n_bad else "")
          + (f"; rebuilt by a step: {n_same} identical, {n_diff} different" if n_same + n_diff else "")
          + (f"; {n_pending} to be regenerated from the raw data by steps 01a/01b" if n_pending else ""))
    return n_bad


def _step_key(v: str, opt: str) -> str:
    """Normalise a stage or step id given on the command line ('1' -> '01', '5a' -> '05a').

    Args:
        v: the value given by the user.
        opt: the option name, for the error message.

    Returns:
        The normalised stage (two digits) or step id. Exits with an error if no step matches.
    """
    m = re.fullmatch(r"(\d{1,2})([a-z]?)", v.strip().lower())
    key = (m.group(1).zfill(2) + m.group(2)) if m else v
    if not m or (m.group(2) and key not in BY_ID) or not any(s.id.startswith(key) for s in STEPS):
        sys.exit(f"{opt} {v!r}: no such stage or step. Valid stages: "
                 f"{', '.join(sorted({s.id[:2] for s in STEPS}))}; steps: {', '.join(BY_ID)}")
    return key


def select(args) -> list[Step]:
    """Return the steps chosen by ``--steps``, or else by ``--from``/``--to``, in run order."""
    if args.steps is not None:
        out = []
        for v in (x for x in args.steps.split(",") if x.strip()):
            key = _step_key(v, "--steps")
            if key not in BY_ID:
                sys.exit(f"--steps {v!r}: give step ids (e.g. 05a), not a stage. Steps: {', '.join(BY_ID)}")
            out.append(BY_ID[key])
    else:
        lo, hi = _step_key(args.from_, "--from"), _step_key(args.to, "--to")
        out = [s for s in STEPS if s.id >= lo and s.id[:len(hi)] <= hi]
    if not out:
        sys.exit("no step selected"
                 + (f" (--steps {args.steps!r})" if args.steps is not None
                    else f" (--from {args.from_} --to {args.to})") + "; see --list")
    return out


def stale_pooled_oof() -> str | None:
    """Tell whether step 07a would fit on a stale ``artifacts/mrft_oof_3fold.csv``.

    After a re-run of 06c, that file must be re-pooled from the new qual files first.

    Returns:
        None if 07a may run (06c was not re-run, or the pooled file is newer than its qual files);
        otherwise the reason it must not.
    """
    qual = [ROOT / f"{DKT}/results/qual_lmkt_mrft_sa_8b_a1f{k}_incfirst_{k}.csv" for k in (1, 2, 3)]
    qual = [q for q in qual if q.exists()]
    if not qual:
        return None                             # 06c not re-run: the shipped pooled file is used
    pooled = ROOT / "artifacts/mrft_oof_3fold.csv"
    if not pooled.exists():
        return ("artifacts/mrft_oof_3fold.csv is missing (06c moved the shipped copy to "
                "artifacts/_bundle/): " + STOP_AFTER["06c"])
    if max(q.stat().st_mtime for q in qual) > pooled.stat().st_mtime:
        return ("the 06c qual files are newer than artifacts/mrft_oof_3fold.csv, so 07a would fit "
                "on out-of-fold predictions of the old adapters: " + STOP_AFTER["06c"])
    return None


def step_env(env: dict, step: Step) -> dict:
    """Return the environment a step's commands run in.

    That is the run's environment plus the step's own variables (``ANNO_ENV`` for 02*). For the
    vLLM steps (02*), ``VLLM_WORKER_MULTIPROC_METHOD`` defaults to ``spawn``: without it vLLM can
    abort with "Cannot re-initialize CUDA in forked subprocess".
    ``VLLM_USE_FLASHINFER_SAMPLER`` is left alone, because the production 02a run used the
    FlashInfer sampler.

    Args:
        env: the run's environment.
        step: the step.

    Returns:
        A new environment dict.
    """
    out = {**env, **step.env}
    if step.id.startswith("02"):
        out.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return out


def main() -> None:
    """List, check, verify or run the selected steps (see the module docstring).

    Returns:
        None. Each step writes its declared outputs into the workspace. Exits 0 on success, 1 on a
        failed check or step, and 3 when the run stops after 02a or 06c with selected steps unrun.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list every step and exit")
    ap.add_argument("--check", action="store_true",
                    help="check the CPU environment and the data (with --with-gpu, also the GPU "
                         "interpreters and cached base models of the selected steps)")
    ap.add_argument("--verify", action="store_true",
                    help="sha256 of every shipped intermediate file; rebuilt files listed separately")
    ap.add_argument("--from", dest="from_", default="01", help="first stage or step (e.g. 03, 02b)")
    ap.add_argument("--to", default="09", help="last stage or step (e.g. 09, 06c)")
    ap.add_argument("--steps", default=None, help="comma list of step ids, e.g. 05a,07a (run in that order)")
    ap.add_argument("--with-gpu", action="store_true",
                    help="also run the GPU steps (02*, 06*); stops after 02a and 06c for manual steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every selected step's commands without running them (through "
                         "src/run_training.py the workspace is still assembled or refreshed "
                         "first: the skip notes read it)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # step headers appear in order when piped to a log
    manifest = load_manifest()

    if args.list:
        for s in STEPS:
            print(f"{s.id}  [{s.kind}] {s.title}  ({s.est}; expect: {s.expect})")
        return
    interp = interpreter_problems()
    if args.check:
        bad = check_environment()
        for p in interp:
            print(f"  ERROR: {p}")
        bad += len(interp)
        if args.with_gpu:
            problems = gpu_preflight(select(args))
            for p in problems:
                print(f"  GPU: {p}")
            print(f"  GPU interpreters and base models: {'ok' if not problems else 'see README, Setup'}")
            bad += len(problems)
        sys.exit(1 if bad else 0)
    if args.verify:
        sys.exit(1 if verify_bundle(manifest) else 0)

    steps = select(args)
    if interp:
        msg = "\n".join(f"  {p}" for p in interp)
        if not args.dry_run:
            sys.exit(f"refusing to start:\n{msg}")
        print(f"WARN (dry run): a real run would refuse to start:\n{msg}")
    if args.with_gpu:
        problems = gpu_preflight(steps)
        if problems:
            msg = "\n".join(f"  {p}" for p in problems)
            if not args.dry_run:
                sys.exit(f"--with-gpu: the interpreters or base models are not ready (README, "
                         f"Setup and the headers of requirements-*.txt):\n{msg}")
            print(f"WARN (dry run): a real run with --with-gpu would refuse to start:\n{msg}")
    if not args.dry_run:
        (ROOT / "figures/data").mkdir(parents=True, exist_ok=True)   # report scripts write here
    env = dict(os.environ, MAGNIFICAT=str(ROOT), PYTHONHASHSEED="0")
    for i, s in enumerate(steps):
        print(f"\n=== {s.id} {s.title}  [{s.kind}, {s.est}]")
        if s.note:
            print(f"    note: {s.note}")
        senv = step_env(env, s)
        added = {k: v for k, v in senv.items() if env.get(k) != v}
        if added:
            print("    env: " + " ".join(f"{k}={v}" for k, v in sorted(added.items())))
        skip = None
        if s.kind == "gpu" and not args.with_gpu:
            have = s.provided or s.outputs
            missing = [o for o in have if not (ROOT / o).exists() and not (BUNDLE_BACKUP / o).exists()]
            skip = ("GPU step; its outputs ship in the repository; "
                    + (f"present: {', '.join(Path(o).name for o in have)}" if not missing
                       else f"MISSING: {missing}"))
        elif absent := [r for r in s.requires if not (ROOT / r).exists()]:
            skip = (f"its inputs are produced by a GPU step that was not re-run: "
                    f"{', '.join(Path(r).name for r in absent)}; the shipped output is used instead")
        if args.dry_run:
            for c in s.cmds:
                print(f"    $ {c}")
            if skip:
                print(f"    (would be skipped: {skip})")
            elif s.id == "07a" and (why := stale_pooled_oof()):
                print(f"    (would refuse to run: {why})")
            elif args.with_gpu and s.id in STOP_AFTER:
                print(f"    (the run would stop after this step: "
                      f"{stop_message(s.id, [t.id for t in steps[i + 1:]])})")
            continue
        if skip:
            print(f"    SKIPPED ({skip})")
            continue
        if s.id == "07a" and (why := stale_pooled_oof()):
            sys.exit(f"step 07a refused: {why}")
        for c in s.cmds:
            print(f"    $ {c}")
        backup_bundled(s.outputs + s.backup)
        t0 = time.time()
        for c in s.cmds:
            rc = subprocess.call(["bash", "-c", c], cwd=ROOT, env=senv)
            if rc != 0:
                sys.exit(f"step {s.id} failed (rc={rc}) on: {shlex.quote(c)}")
        print(f"    done in {time.time() - t0:.0f}s")
        if n_missing := report_outputs(s, manifest):
            sys.exit(f"step {s.id} exited 0 but {n_missing} of its outputs are MISSING (listed "
                     "above); fix that before running the steps that read them")
        if s.id in STOP_AFTER:
            rest = [t.id for t in steps[i + 1:]]
            print(f"\nSTOPPED after {s.id}: manual step needed. Now {stop_message(s.id, rest)}"
                  + (f"\n(not run: {', '.join(rest)})" if rest else ""))
            sys.exit(3 if rest else 0)


if __name__ == "__main__":
    main()
