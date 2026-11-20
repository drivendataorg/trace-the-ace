# src/training/launchers

Scheduler scripts. None of them is needed to use or rebuild the model: the training entry point is
`python src/run_training.py` (step list in `src/training/reproduce.py`), and the GPU steps are
described in the README's [Run training](../../../README.md#run-training) section (Stages 02 and 06).

| folder | what it is | runs here? |
|---|---|---|
| `original/` | the cluster and UCloud launchers we kept of the runs that produced the shipped files (commands verbatim; comments shortened), as a record of scheduler, GPU, environment-variable and environment settings (not every run's launcher was kept; see below) | no: they hard-code our hosts' paths (`/dtu-compute/…`, `/work/…`) and queue names |
| `release/` | `stage_bundle.py`, the one-off script that collected the intermediate files from our research host into this repository's layout (ran once, 2026-10-17) | no: reads our research host's paths; kept for provenance |
| `gpu_noise/` | the LSF launchers of the GPU-noise study: `submit_all.sh` submits `score.bsub` once per GPU type | yes, on an IBM LSF cluster, after replacing the queue names and `select[]` resources with your cluster's |

## original/

- `dialogue-kt/lsf/`: DTU HPC (LSF) jobs of the LoRA fork: stage 1 (`traceace_qwen3_8b_full_final`),
  the three fold parents (`traceace_qwen3_8b_full_fold{1,2,3}`), the MRBench annotation job,
  the MRBench continuation of fold parent 1 and its held-out scoring, the NTO annotation and the
  competition-table build.
- `dialogue-kt/ucloud_scripts/anno_8gpu.sh`: the 8-GPU UCloud annotation launcher.
- `magnificat/lsf/`: DTU HPC jobs of the student-state annotation and of the feature, distillation
  and XGBoost steps (`v2ab/` holds the taxonomy-v2 distillation; `v2fam/` holds the taxonomy-v2
  session block, the v2shrunkD build, the submission-zip build and the final smoke test).
- `ucloud/`: the UCloud B200 scripts of LoRA stage 2 (`run_mrft_all35k.sh` = step 06b, the shipped
  adapter; `run_mrft_f2.sh`, `run_mrft_f3.sh` = the fold-2/3 continuations of 06c) and
  `setup_env.sh`, the setup script `src/training/requirements-lora-stage2.txt` was reconstructed
  from.

Not kept: the launchers of the 02a merge (the re-run of failed chunks), of 02d and of 02e, and
the pooling of the 06c fold predictions. Step 07a needs no launcher: it is a CPU step that
`reproduce.py` runs.

`dialogue-kt/lsf/mr_train_8b.bsub` ends by writing a `LICENSE_GATE.txt` into the adapter folder
that calls the weights shippable only after a share-alike check. That file is not a licence: the
adapters trained on MRBench are released under CC BY-SA 4.0 (see
[models/llm/NOTICE.md](../../../models/llm/NOTICE.md)).

## gpu_noise/

`submit_all.sh` finds the repository root from its own location and writes results and job logs to a configured location. Set `PY` to the Python 3.12 environment of
[README Setup](../../../README.md#setup) step 2 and, if the base model is not in
`models/huggingface_models/Qwen/Qwen3-8B`, export `LLMKT_BASE_DIR`. Set `OUT` to write results to
another folder (job logs then go to `$OUT/logs`). `PY`, `LLMKT_BASE_DIR` and `OUT` are written
into each job script, because LSF did not pass exported variables on to our jobs. With `OUT` set,
build the table with
`python src/tools/gpu_noise_report.py --runs $OUT --labels /path/to/test_labels.csv`.
The test files are taken from `data/raw/` (README Setup step 4). On a machine without LSF, run
`src/tools/score_llm_half.py` directly (its docstring gives the command).

Our own runs used an equivalent launcher that hard-codes our paths and has a fourth block, D,
that also runs `src/run_inference.py` end to end on the full test set and scores it with
`src/tools/score_submission.py`. This copy is the same study, parametrised, with blocks A-C.
