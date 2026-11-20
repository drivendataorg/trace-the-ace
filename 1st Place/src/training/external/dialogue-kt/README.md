> This copy is team appelswim's fork of
> [umass-ml4ed/dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt), used for the Trace the
> Ace solution. It is upstream commit `c61f335` plus our changes, listed below. To see every
> change, compare this folder with upstream `c61f335`:
> `git clone https://github.com/umass-ml4ed/dialogue-kt && git -C dialogue-kt checkout c61f335 &&
> diff -ruN -x .git -x data -x results -x saved_models -x .gitignore dialogue-kt <this folder>`
> (the excluded folders hold upstream data and placeholders this fork does not ship). Upstream's
> text below is unchanged except for the "Fork additions" section and the short notes marked
> "Fork note".
>
> - **Added by us:** `anno_json.py`, `annotate_mrbench_vllm.py`, `annotate_nto_moves.py`,
>   `annotate_qatd_vllm.py`, `annotate_student_states.py`, `annotate_traceace_qwen32b.py`,
>   `annotate_traceace_vllm.py`, `build_mrbench_pkl.py`, `build_traceace_kt.py`, `run_kt.sh`,
>   `run_kt_lsf.sh`.
> - **Modified by us:** `dialogue_kt/annotate.py`, `data_loading.py`, `kt_data_loading.py`,
>   `main.py`, `prompting.py`, `training.py`, `models/lm.py`, `LICENSE` (our copyright line added)
>   and this README.
> - **Not included:** upstream's annotated CoMTA and MathDial files and `COMTA_LICENSE.txt` (the
>   "Annotated Data" section below refers to them), and upstream's `.gitignore` and empty
>   `results/` and `saved_models/` placeholders. CoMTA's licence is non-commercial and the
>   Trace the Ace pipeline does not need them. `data/annotated/` is created at run time:
>   `build_traceace_kt.py` writes the competition table there (`traceace.pkl`; the pipeline uses
>   `traceace_full.pkl` via `TRACEACE_PKL`, built in step 01a and not shipped because it holds
>   competition text), `build_mrbench_pkl.py` writes the MRBench tables, and `data_loading.py`
>   reads them from there.
> - **Not for this pipeline:** upstream's `requirements.txt` and its "Environment" section below
>   describe upstream's own setup (transformers 4.44.0, which cannot load Qwen3). Use
>   `src/training/requirements-lora-stage1.txt` (stage 1) and
>   `src/training/requirements-lora-stage2.txt` (stage 2) from the repository root (see the
>   sections "Setup" and "Run training" of the main README). `OPENAI_API_KEY` is not needed:
>   our annotators run open models locally.
> - **Optional switches:** `reproduce.py` and the launchers in
>   `src/training/launchers/original/` set none of `ANNO_BATCH`, `ANNO_DTYPE`, `ANNO_MAXLEN`,
>   `ANNO_MAX_SEQS`, `ANNO_N_DIALOGUES`, `ANNO_N_PARITY`, `ANNO_OUT`, `ANNO_PARITY`, `ANNO_PP`,
>   `ANNO_TAXONOMY`, `ANNO_THINK`, `LLMKT_BLANK_FINAL_TURN`, `LLMKT_COMPILE`,
>   `LLMKT_DATALOADER_WORKERS`, `LLMKT_DEGENERATE_KC`, `LLMKT_FAST_LOSS`, `LLMKT_GRAD_CKPT`,
>   `LLMKT_GROUP_BY_LENGTH`, `LLMKT_NO_DATASET_CONTEXT`, `LLMKT_N_SPLITS`, `LLMKT_TERMINAL_ONLY`,
>   `LLMKT_TEST_SHARD`, `LLMKT_VAL_*`, `SAVE_EVERY_STEPS`, `VAL_EVERY_STEPS` and
>   `VAL_MAX_BATCHES`; the documented steps run with their defaults.
>
> Licence: MIT (`LICENSE`), with two copyright holders: Alexander Scarlatos (2024) for the original
> code and team appelswim (2026) for our additions and modifications.

# Dialogue Knowledge Tracing
This is the official repo for the paper <a href="https://arxiv.org/abs/2409.16490">Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs</a>. The primary contributions here include 1) LLMKT and DKT-Sem, our language model-based KT models, 2) code to train and evaluate KT models, including the DKT family and BKT, on the dialogue knowledge tracing task, and 3) code to automatically annotate dialogues with knowledge component and correctness labels using the OpenAI API.

If you use our code or find this work useful in your research then please cite us!
```
@inproceedings{scarlatos2024exploringknowledgetracingtutorstudent,
      title={Exploring Knowledge Tracing in Tutor-Student Dialogues using LLMs},
      author={Alexander Scarlatos and Ryan S. Baker and Andrew Lan},
      year={2025},
      booktitle={Proceedings of the 15th Learning Analytics and Knowledge Conference, {LAK} 2025, Dublin, Ireland, March 3-7, 2025},
      publisher={{ACM}},
}
```

## Annotated Data

Annotated versions of the CoMTA and MathDial datasets (i.e. including per-turn knowledge component and correctness labels) are available in `data/annotated`, and can be loaded as-is during knowledge tracing training.

These versions of the datasets are subject to their original licenses. The license for CoMTA is available in `data/annotated/COMTA_LICENSE.txt` and MathDial is licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

## Setup

### Download Data
This step is not necessary to reproduce our knowledge tracing results since we release the annotated data in `data/annotated`. However, you can follow the steps below to replicate our workflow or to experiment with custom data annotation.

<b>Achieve the Core (ATC)</b>: Download the <a href="https://huggingface.co/datasets/allenai/achieve-the-core">ATC HuggingFace dataset</a> and put `standards.jsonl` and `domain_groups.json` under `data/src/ATC/`. At the time of releasing this code, the data was not accessible via HuggingFace due to a bug. If the data is still not accessible then you can contact us or the authors of <a href="https://arxiv.org/pdf/2408.04226">the paper</a> to send you a copy.

<b>CoMTA</b>: Download the <a href="https://github.com/Khan/tutoring-accuracy-dataset/blob/main/CoMTA_dataset.json">CoMTA data file</a> and put it under `data/src`.

<b>MathDial</b>: Clone the <a href="https://github.com/eth-nlped/mathdial/tree/main">MathDial repo</a> and put the root under `data/src`.

### Environment
> Fork note: this section is upstream's setup (see the top of this README for the environments
> we use). In particular, the
> `OPENAI_API_KEY=<your key here>` line below is upstream's placeholder for its OpenAI-based
> annotation; our annotators run open models locally and need no key.

We used Python 3.10.12 in the development of this work. Run the following to set up a Python environment:
```
python -m venv dk
source dk/bin/activate
pip install -r requirements.txt
```

Also add the following to your environment:
```
export OPENAI_API_KEY=<your key here> # For automated annotation via OpenAI
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # For enabling deterministic operations
```

### Fork additions: training log and optional knobs (env vars)

`train_lmkt` writes `saved_models/<model_name>/train_log.csv` with the schema
`run,epoch,opt_step,split,loss,n`. `split` is `train` (per optimizer step),
`train_epoch` (epoch mean) or `val`. `run` increments on every relaunch: a crashed
attempt leaves its rows behind and the restart appends to the same file, so without it
the curves of different attempts would run together.

| var | default | effect |
| --- | --- | --- |
| `SAVE_EVERY_STEPS` | `0` | save the adapter every N optimizer steps, not batches (DDP divides `grad_accum` by the number of ranks, so an epoch has far fewer optimizer steps than batches); written to `<checkpoint>_midepoch` |
| `VAL_EVERY_STEPS` | `0` | score the monitor set every N optimizer steps and append a `split=val` row; `0` = epoch end only |
| `VAL_MAX_BATCHES` | `0` (all) | cap the batches per validation probe to keep it cheap |
| `LLMKT_VAL_PKL` | unset | pkl to use as the monitor set; needed for a validation curve under `--train_all`, which otherwise leaves no hold-out |
| `LLMKT_VAL_FOLD` | the run's `--fold`, else `1` | which LO-disjoint block of that pkl to score; `all` = every row (for OOD pkls like QATD) |
| `LLMKT_VAL_MAX` | `0` (all) | subsample the monitor set to at most N rows |
| `LLMKT_FAST_LOSS` | unset | vectorized unpacked loss (bit-identical for `mean-ar`/`prod`) |
| `LLMKT_DATALOADER_WORKERS` | `0` | tokenize on separate cores (unpacked path only) |
| `LLMKT_GROUP_BY_LENGTH` | unset | length-bucketed DDP sampler |

## Prepare Dialogues for KT (Run Annotation with OpenAI)
> Fork note: upstream's annotation of CoMTA and MathDial with the OpenAI API, kept for reference.
> This repository annotates with open models via vLLM (`annotate_traceace_vllm.py` and the other
> `annotate_*.py` scripts in this folder).

This step is not necessary to reproduce our results because we release the annotated datasets, but is here for reference.

Dialogue KT requires each dialogue turn to be annotated with correctness and knowledge component (KC) labels. We automate this process with LLM prompting via the OpenAI API. You can run the following to tag correctness and ATC standard KCs on the two datasets:
```
python -m dialogue_kt.main annotate --mode collect --openai_model gpt-4o --dataset comta
python -m dialogue_kt.main annotate --mode collect --openai_model gpt-4o --dataset mathdial
```

To see statistics on the resulting labels, run:
```
python -m dialogue_kt.main annotate --mode analyze --dataset comta
python -m dialogue_kt.main annotate --mode analyze --dataset mathdial
```

## Train and Evaluate KT Methods
Each of the following runs a train/test cross-validation on the CoMTA data for a different model:
```
python -m dialogue_kt.main train --dataset comta --crossval --model_type lmkt --model_name lmkt_comta         # LLMKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkt-sem --model_name dkt-sem_comta   # DKT-Sem
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkt --model_name dkt_comta           # DKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type dkvmn --model_name dkvmn_comta       # DKVMN
python -m dialogue_kt.main train --dataset comta --crossval --model_type akt --model_name akt_comta           # AKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type saint --model_name saint_comta       # SAINT
python -m dialogue_kt.main train --dataset comta --crossval --model_type simplekt --model_name simplekt_comta # simpleKT
python -m dialogue_kt.main train --dataset comta --crossval --model_type bkt                                  # BKT
```

Check the `results` folder for metric summaries and turn-level predictions for analysis.

To see all training options, run:
```
python -m dialogue_kt.main train --help
```

### Hyperparameter Sweep
We run a grid search to find the optimal hyperparameters for the DKT family models. For example, to run a search for DKT on CoMTA, run the following (crossval is inferred and model_name is set automatically):
```
python -m dialogue_kt.main train --dataset comta --hyperparam_sweep --model_type dkt
```

The output will indicate the model that achieved the highest validation AUC. To get its performance on the test folds, run:
```
python -m dialogue_kt.main test --dataset comta --crossval --model_type dkt --model_name <copy from output> --emb_size <get from model_name>
```

#### Best Hyperparameters Found

CoMTA:
- DKT-Sem: lr=2e-4, emb_size=256
- DKT: lr=1e-3, emb_size=32
- DKVMN: lr=1e-4, emb_size=16
- AKT: lr=5e-3, emb_size=32
- SAINT: lr=1e-3, emb_size=32
- simpleKT: lr=2e-4, emb_size=16

MathDial:
- DKT-Sem: lr=2e-3, emb_size=512
- DKT: lr=5e-3, emb_size=256
- DKVMN: lr=1e-3, emb_size=128
- AKT: lr=2e-4, emb_size=64
- SAINT: lr=2e-4, emb_size=64
- simpleKT: lr=5e-4, emb_size=256

## Visualize Learning Curves
To generate the learning curve graphs, run the following (they will be placed in `results`):
```
python -m dialogue_kt.main visualize --dataset comta --model_name <trained model to visualize predictions for>
```
