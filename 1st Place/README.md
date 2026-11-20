# Solution - Trace the Ace

Username: **appelswim** (team: Louis Boucherie, Tereza Blazkova).

**TL;DR.** This is our solution to Trace the Ace (private leaderboard log loss 0.59241). The model
averages two calibrated predictions: a LoRA fine-tune of Qwen3-8B that takes the whole transcript as
input (weight 0.57) and an XGBoost model on features computed from the transcript (weight 0.43). It
comes as two zips. `trace-the-ace_prize-winner.zip` is the full solution from our original training
runs: the code, every model from those runs, the scored submission (`models/submission.zip`, same
code and predictions), all GPU outputs and the intermediate files, so the CPU part of training can
be rebuilt without a GPU. `trace-the-ace_prize-winner_seed-replicates.zip` (optional, 2.1 GB) holds
the LLM half retrained with four more seeds (222 to 225; the shipped one used 221). Neither
contains competition data. The rest of this README
explains how the model works and how each stage scores, how the folders are laid out, and how to set
up, retrain and run the model; the licences are at the end. AUROC is written × 100 (64.73 means
0.6473).

Competition: [Trace the Ace](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/)
on the K-12 AI Infrastructure Platform (run by DrivenData, for Digital Promise and the National
Tutoring Observatory). Given a 1:1 student–tutor session transcript and a learning objective (LO),
predict the probability that the student then answers the next question on that objective
correctly. Metric: log loss.

| | log loss | AUROC |
|---|---:|---:|
| public leaderboard | 0.5945 | 64.77 |
| **private leaderboard (final)** | **0.59241** | **64.735** |
| all 10,508 released test rows, scored after the competition | 0.5934 | 64.73 |

The public and private sets are disjoint parts of the released test rows, and which rows were public
was not released, so the private 0.59241 cannot be re-scored on its own. The third row is our own
re-run of the same model on all of them. Log loss is a per-row mean, so the full-test score is
f · public + (1 − f) · private. Our 0.59345 is consistent with the board at an implied public share
f ≈ 0.50 (0.48–0.51 after rounding), the usual 50/50 split. This is a consistency check, not a
re-score. AUROC is not a row mean and cannot be checked this way; our full-test 64.73 is close to
both board values. GPU arithmetic moves the full-test log loss with sd 0.00004 across eight
full-test runs on six GPU models.

**Contents:** [Summary](#summary) · [Scorecard](#scorecard-the-score-of-every-stage) ·
[Repository organization](#repository-organization) · [Setup](#setup) · [Hardware](#hardware) ·
[Run training](#run-training) · [Run inference](#run-inference) ·
[Data and licences](#data-and-licences)

**What you can do with this repository** (every command runs from the repository root)

| goal | how | time |
|---|---|---|
| try the model on a one-row example | `make example` ([Run inference](#run-inference)) | ~4 min on a 6-core CPU |
| predict the competition test set, or new data | `python src/run_inference.py` ([Run inference](#run-inference)) | 10,508 rows: 1 h 54 min on an H100 PCIe and 2 h 38 min on an A100 80 GB (measured); about 3 h 38 min projected on the platform's A100 |
| submit the model to DrivenData again | upload `models/submission.zip` on the competition's code-execution page ([Run inference](#run-inference)) | upload |
| check that the shipped intermediate files rebuild the submission | `make quick-check` ([Run training](#run-training)) | ~15 min (12 min measured), CPU only |
| rebuild the CPU parts of the model from the raw data (the GPU outputs ship) and check every stage's score | `python src/run_training.py --from 01 --to 09` ([Run training](#run-training)) | ~4 h, CPU only (3 h 50 min measured) |
| retrain the GPU parts too (LLM annotation, LoRA fine-tuning) | [Retraining the GPU steps](#retraining-the-gpu-steps) | ~275 GPU-hours |

**No competition data is included.** The repository holds our code, the trained models, our own
LLM annotations (ids and labels only), derived feature tables, and MRBench (an external dataset
used in training, CC BY-SA 4.0). Out-of-fold predictions ship without the training labels, which
`src/make_workspace.py` restores from `train_labels.csv` byte for byte; the two intermediate files
that consist of transcript text are regenerated from the raw data. As in any trained text model,
the TF-IDF vocabularies inside the trained classifiers contain single words, word pairs and short
character n-grams (3–5 characters) from the training transcripts and objective texts; they are
stored in sorted order, so the order of the training text is not preserved. The repository also
holds per-row predictions (no labels) on the released test set. See
[Data and licences](#data-and-licences).

---

## Summary

**The model** is a fixed blend of two calibrated models, `p = 0.57 · p_LLM + 0.43 · p_GBM`.

```
             +-----------------------------------------------+
transcript --| LLM half:  Qwen3-8B + LoRA (r=16)             |-- p_llm --+
   + LO      |            -> frozen Platt calibrator         |           |  p = 0.57*p_llm
             +-----------------------------------------------+           +-  + 0.43*p_gbm
             +-----------------------------------------------+           |
transcript --| XGBoost:   177 numeric columns                |-- p_gbm --+
   + LO      |            + LO TF-IDF + LO id one-hot        |
             |            -> frozen Platt (inside the model) |
             +-----------------------------------------------+
```

- **Preprocessing.** For the LLM, each transcript becomes a list of tutor/student turns
  (`background` noise turns are dropped), and the whole transcript plus the objective text form one
  prompt of up to 8,192 tokens. At inference, longer prompts keep the 122-token instruction and the
  last 8,070 tokens (`LLMKT_KEEP_HEADER=1`, set by `main.py`); the LoRA was trained with a plain
  left cut, which keeps the last 8,192 tokens.
  The LLM half's calibrator was also fit on predictions made with the plain cut and was not refit
  for the header-preserving one.
  For the XGBoost half, every student turn is paired with the tutor turns before and after it.
- **LLM half.** A LoRA fine-tune of `Qwen/Qwen3-8B` in the LLMKT formulation (LLM-based knowledge
  tracing; Scarlatos, Baker and Lan, "Exploring Knowledge Tracing in Tutor-Student Dialogues using
  LLMs", LAK 2025, [arXiv 2409.16490](https://arxiv.org/abs/2409.16490)) as implemented in
  [dialogue-kt](https://github.com/umass-ml4ed/dialogue-kt): the model is asked whether
  the student knows the objective, and the prediction is `softmax([logit_True, logit_False])[0]`
  from one forward pass. Stage 1 trains on all 35,062 competition dialogues (2 epochs); stage 2
  continues for 17 optimiser steps on the public MRBench tutoring corpus.
- **XGBoost half.** A strongly regularised XGBoost model (5 seeds) over the objective text (TF-IDF),
  the objective id (one-hot) and 177 transcript features: 54 handcrafted (turn balance, pacing,
  objective/transcript overlap, regex tutor moves, curriculum level), 17 from two regex state
  machines (contingency and repair arcs), and 106 from **distillation**. For distillation, large
  open LLMs (Qwen3.5-122B, Qwen3-32B, Qwen2.5-14B) labelled 0.3–2.0 million individual turns against
  four codebooks (answer correctness, student states, our second tutor/student move codebook
  "taxonomy v2", and an interview codebook built from conversations with three practising tutors).
  Each set of labels was distilled into a TF-IDF + logistic-regression classifier that runs on CPU
  (five classifiers: taxonomy v2 has one for tutor and one for student moves), and its per-turn
  probabilities are aggregated into session rates, shrunk towards the corpus mean for short
  sessions. A sixth classifier (NTO tutor moves) still runs at inference, but its 21 columns were
  dropped before training and the model ignores them.
- **Ensemble and output.** Each half has its own frozen Platt calibration fitted on out-of-fold
  training predictions; the calibrated probabilities are averaged with a frozen weight (0.43 for
  the XGBoost half) and clipped to [1e-6, 1 − 1e-6]. There is no post-blend calibration and no use
  of test statistics.
- **Validation.** Out-of-fold predictions use `GroupKFold` by learning objective (no objective in
  both the training and the held-out part), because the test set contains objectives unseen in
  training; folds that share objectives would let a model memorise per-objective base rates. The
  fold partition is frozen (sha `59f0c807dad3ac08`). Scores are also reported on rows whose
  session is unseen, always as log loss and AUROC together, and no feature is a per-objective
  statistic of the labels.
- **Independence of test rows.** No prediction depends on another row. `main.py` still computes
  four session-level task columns by grouping the test rows that share a transcript, but the
  shipped XGBoost model was trained without them and its frozen column selection discards them.

### Scorecard: the score of every stage

Each cell: log loss / AUROC. The row sets:

- *Out-of-fold (16,693)*: training rows predicted by an LLM fold model that never saw their
  objective, with the XGBoost half's own objective-disjoint out-of-fold prediction for the same
  rows. The LLM fold models cover folds 1–3 of dialogue-kt's own 5-fold objective-disjoint split
  (7,273 + 4,371 + 5,049 = 16,693 rows).
- *XGBoost-clean (16,687)*: rows whose objective and session the XGBoost fold model did not see
  (the shipped mask `data/interim/folds/clean_rows_gkf5.npy`). Stage 05 reports on this set.
- *Clean (7,738)*: XGBoost-clean ∩ the 16,693 out-of-fold rows. The LLM fold splits are
  objective-disjoint but not session-disjoint, so these rows are session-disjoint for the XGBoost
  half only.
- *Released test (10,508)*: the released test rows, scored with DrivenData's post-competition
  labels.

| component | out-of-fold (16,693) | clean (7,738) | released test (10,508) |
|---|---:|---:|---:|
| constant = training base rate (0.7025) | 0.6059 / 50.0 | 0.6657 / 50.0 | 0.6229 / 50.0 |
| **XGBoost half** (177 cols incl. distilled features) | 0.5661 / 67.32 | 0.6233 / 65.17 | 0.6015 / 62.55 |
| **LLM half** (LoRA stage 1 + 17 steps on MRBench), raw | 0.5826 / 67.56 | 0.6396 / 65.53 | – |
| LLM half, calibrated (shipped) | 0.5653 / 67.56 | 0.6232 / 65.53 | 0.5973 / 64.03 |
| **ensemble = the submitted model** | **0.5596 / 68.99** | **0.6166 / 66.86** | **0.5934 / 64.73** |

All calibrators use training data only. Outside the test column, the calibrated cells are
slightly optimistic: in the out-of-fold and clean columns, every calibrated number (the calibrated
LLM half, the XGBoost half and the ensemble) uses the shipped Platt maps, which were fit on training
out-of-fold predictions that include these rows, and the blend weight was set on out-of-fold
rows. Only the raw row is free of this, and only the released-test column is fully held out. The
test cells come from the shipped `main.py` (LSF job 29267305), whose outputs are calibrated, so
the raw row has no test cell.

The two halves are about equally good and make different mistakes: on clean rows the LLM
(0.6232) and the XGBoost half (0.6233) tie and the blend gains 0.0066 over either.

Every cell can be recomputed. After Setup and with the training files in `data/raw/`:

```bash
python src/make_workspace.py                  # assembles build/workspace/ and restores the OOF labels
cd build/workspace && python scripts/score_stages.py
python scripts/score_stages.py --test-labels /path/to/test_labels.csv \
  --test-components ../../models/predictions/test/shipped_components.csv
cd ../..                                      # back to the repository root
```

The output goes to `build/workspace/figures/data/scorecard.md` and should match the table above.
`test_labels.csv` is DrivenData's post-competition release; it is not included. The training
steps below each print their own scores.

**The distilled classifiers** (quality = mean out-of-fold AUROC against the annotator's labels):

| classifier → XGBoost block | annotator | labelled turns / sessions | heads | out-of-fold AUROC |
|---|---|---|---:|---:|
| answer correctness → `recdst_` (14 cols) | Qwen3.5-122B-A10B-FP8 | 2,038,332 / 22,803 | 2 | is-an-answer 82.53 · is-correct 73.04 |
| NTO tutor moves → `ntodst_` (21 cols, computed at inference but removed before training with `--drop-cols ntodst_`, so the booster ignores them) | Qwen3-32B | 2,364,199 / 22,811 | 19 | 82.19 |
| student states → `studst_` (12) | Qwen3-32B | 497,121 / 4,800 | 12 | 84.86 |
| interview codebook → `d29dst_` (34) | Qwen2.5-14B-Instruct | 1,026,686 records / 4,000 | 14 | 90.80 |
| taxonomy-v2 tutor moves → `v2ht_` (34) | Qwen3.5-122B-A10B-FP8 | 316,225 / 3,072 | 32 | 88.45 |
| taxonomy-v2 student moves → `v2hs_` (12) | Qwen3.5-122B-A10B-FP8 | 316,772 / 3,072 | 12 | 87.09 |

These AUROCs measure how faithfully each classifier copies its annotator, not whether the labels
are right: every label set comes from an LLM. For answer correctness we checked the annotators on
MathDial. Against GPT-4o's labels on 300 dialogues, Qwen3.5-122B matches 0.919 of the binary
labels (κ 0.693; 3-way agreement 0.815) and Qwen3-32B 0.890 (κ 0.729; 3-way agreement 0.833);
both are agreement with another LLM. Against
teacher-verified solution steps, Qwen3.5-122B reaches κ 0.639 and Qwen3-32B κ 0.515; the 32B model
is lenient (it calls 76.7 % of steps correct where teachers say 63.0 %), and it produced the MRBench
labels of LoRA stage 2. The student-state labels have no human check. The interview-codebook and
taxonomy-v2 labels had only a spot check by one of us, with no inter-rater reliability.

**Names used in file names.** `lmkt_traceace_qwen3_8b_full_final` = LoRA stage 1, `_full_{1,2,3}` =
its out-of-fold twins; `mrft` = the MRBench continuation (stage 2), `all35k` = trained on all
training rows (shipped), `a1f1` = fold-1 twin; `a1` = stage 1; `sub0808` = the shipped calibrator
of the LLM half; `122bC` = built from the Qwen3.5-122B taxonomy-v2 labels; `v2ext` / `v2stu` =
taxonomy-v2 tutor / student labels; `v2shrunkD` = the shipped XGBoost model (with the κ-shrunk
v2 block); `doc29` = the interview codebook; `keephdr` = header-preserving truncation. One line
per file:
[models/README.md](models/README.md).

### Repository organization

```
.
├── README.md                    <- this file
├── LICENSE                      <- MIT for our work; scope and exceptions at the end of the file
├── THIRD_PARTY_NOTICES.md       <- licences and attributions of the third-party material we ship
├── requirements.txt             <- inference packages (runtime pins of 2026-06-15; see its header)
├── Makefile                     <- shortcuts for the commands below (run `make help` for all)
├── MANIFEST.sha256              <- sha256 of every file in the main archive (Setup step 0 checks it)
├── .gitattributes               <- Git LFS rules, used only if the package is put into git
├── data
│   ├── raw/                     <- the competition data goes here (not included; see raw/README.md)
│   ├── external/                <- external data used in training (NOTICE.md: sources, licences)
│   │   ├── mrbench/raw/         <-   MRBench V1, V2, V3 dev/test JSON, unmodified (CC BY-SA 4.0)
│   │   ├── mrbench/derived/     <-   our correctness labels for MRBench + the stage-2 training tables
│   │   └── nc_curriculum.csv    <-   National Curriculum in England statements (OGL v3.0)
│   ├── interim/                 <- our intermediate data, ~1.0 GB (no competition text or labels)
│   │   ├── annotations/         <-   LLM turn labels: session id, turn number, label
│   │   ├── turn_probabilities/  <-   per-turn probabilities of the distilled classifiers
│   │   ├── session_features/    <-   the session-level feature tables the XGBoost model trains on
│   │   └── folds/               <-   the frozen objective-disjoint folds and the XGBoost-clean row mask (16,687)
│   ├── examples/new_data/       <- a synthetic one-row example for inference on new data
│   └── processed/               <- inference writes submission.csv here
├── models                       <- every trained weight, ~1.7 GB (README.md: one line per file)
│   ├── submission/              <- the submission tree: main.py + assets/, ~290 MB (what inference loads)
│   ├── submission.zip           <- the same tree zipped, 263 MB (Run inference: how it relates to the scored zip)
│   ├── scored_zip_replaced_entries.tsv <- entry-by-entry comparison of submission.zip with the scored zip
│   ├── llm/                     <- six LoRA adapters, ~175 MB each: stage 1, stage 2 (shipped), out-of-fold twins;
│   │                               seeds/ = 12 seed replicates, 2.1 GB, separate archive
│   ├── distilled/               <- the 6 distilled turn classifiers, 96 MB
│   ├── gbm/                     <- the XGBoost half (5 seeds + Platt), 4.6 MB
│   ├── calibrators/             <- the frozen Platt map of the LLM half
│   ├── predictions/oof/         <- out-of-fold predictions (training labels removed)
│   ├── predictions/test/        <- per-row predictions on the released test set, no labels (engines: see
│   │                               Scorecard)
│   └── huggingface_models/      <- you download Qwen/Qwen3-8B here (Setup step 3)
├── src
│   ├── run_inference.py         <- inference entry point (runs models/submission/main.py unchanged)
│   ├── run_training.py          <- training entry point (every step, in order, with output checks)
│   ├── make_workspace.py        <- assembles the pipeline's working directory build/workspace/
│   ├── training/                <- the training pipeline
│   │   ├── reproduce.py         <-   the step list: command, environment, outputs, expected result
│   │   ├── workspace_map.tsv    <-   where every intermediate file lives, with its sha256
│   │   ├── scripts/, src/       <-   annotation, distillation, features, XGBoost, calibration, scoring
│   │   ├── external/dialogue-kt/ <-  our fork of dialogue-kt (LoRA training, MIT)
│   │   ├── docs/                <-   the codebooks the annotation prompts are built from
│   │   ├── launchers/           <-   cluster launchers (launchers/README.md): original/ = the launchers
│   │   │                             we kept from the shipped runs (historical), release/ = one-off
│   │   │                             bundle stager (historical)
│   │   └── requirements-*.txt   <-   the four training environments (cpu, lora-stage1: freezes; vllm:
│   │                                 a freeze plus pyarrow; lora-stage2: rebuilt from its setup
│   │                                 script, 4 minor packages unpinned; see each header)
│   └── tools/                   <- packaging, checks, scoring, and data audit
```

---

# Setup

0. **Get the repository.** It comes as a zip archive, `trace-the-ace_prize-winner.zip`, with a
   small checksum file, `trace-the-ace_prize-winner.zip.sha256`. A second archive,
   `trace-the-ace_prize-winner_seed-replicates.zip` (with its own `.sha256`), is optional: it holds
   the LLM half retrained with four more seeds (222 to 225; the shipped LLM half used 221).
   The checksum files are
   published next to the archives or in the e-mail that delivers them, never inside an archive. If
   you received only a checksum line, save it as the archive's name plus `.sha256` next to the
   zip. Put all the files in the same folder, then check the archive, unpack it and check every file:

   ```bash
   sha256sum -c trace-the-ace_prize-winner.zip.sha256      # prints "trace-the-ace_prize-winner.zip: OK"
   unzip -q trace-the-ace_prize-winner.zip && cd trace-the-ace_prize-winner
   sha256sum -c --quiet MANIFEST.sha256 && echo REPOSITORY INTACT
   ```

   To add the seed replicates (they unpack into the same folder):

   ```bash
   cd ..                                                       # the folder that holds the zips
   sha256sum -c trace-the-ace_prize-winner_seed-replicates.zip.sha256
   unzip -q trace-the-ace_prize-winner_seed-replicates.zip     # adds trace-the-ace_prize-winner/models/llm/seeds/
   cd trace-the-ace_prize-winner && sha256sum -c --quiet models/llm/seeds/MANIFEST.sha256 && echo SEED REPLICATES INTACT
   ```

   The main archive unpacks into the single folder `trace-the-ace_prize-winner/`, about 2.7 GB; the
   seed replicates add 2.1 GB in `models/llm/seeds/`. Each archive holds exactly the files listed
   in its manifest (`MANIFEST.sha256`, `models/llm/seeds/MANIFEST.sha256`), plus the manifest
   itself: no `build/` and nothing from `data/raw/` but its README. The archives are built
   reproducibly, so every file inside carries the same fixed date (1980-01-01 by default) and mode
   755 (scripts) or 644 (everything else); that is expected.

   Run every command in this README from the repository root, the folder `trace-the-ace_prize-winner/`.

1. **Prerequisites.** We set up and ran everything on Linux:

   ```
   OS      Linux x86_64 with GNU coreutils/findutils. Tested on Debian 12 (bookworm), and on the
           DTU HPC GPU nodes (AlmaLinux 9.8, recorded by the 2026-10-19 verification jobs) and
           the UCloud B200 nodes of our October 2026 jobs (Ubuntu 24.04). macOS and Windows are
           not supported (the shell tools use GNU find, xargs and realpath).
   tools   sudo apt install curl unzip coreutils make      # Debian/Ubuntu
           sudo dnf install curl unzip coreutils make      # RHEL/Fedora
   Python  3.12.13 for inference and every CPU training step;
           3.10.20 for the byte-identical MATCH lines of steps 01a and 07a (quick check, full
           rebuild) and for GPU stage-1 training. Both are installed with uv, below. Which
           files come out byte-identical also depends on the CPU and on the exact interpreter
           build, and retraining our distilled classifiers exactly needs AVX-512: see
           "Byte-identical or value-identical" under Run training.
   GPU     inference: an NVIDIA GPU with >= 40 GB (details under Hardware). Driver R570 or newer
           recommended (native CUDA 12.8; torch 2.11 ships its own CUDA 12.8 runtime). Older
           CUDA 12 drivers can work through CUDA minor-version compatibility: our workstation's
           driver 535.309.01 ran the LLM half on an RTX A5000. Training on CPU needs no GPU.
   RAM     32 GB recommended. Inference peaked at 15.5 GiB host RAM (maximum resident set) through
           src/run_inference.py on an H100 and an A100, and at ~16 GB on a CPU run (the bf16 base model is loaded and the LoRA merged on CPU on both paths); the
           XGBoost training step peaks at 10 GB.
   disk    repository ~2.7 GB unpacked (models/ ~1.7 GB), 2.1 GB more with the optional seed
           replicates, plus the zips until you delete them; base model 16.4 GB; Python
           environments ~8 GB (.venv) plus ~6 GB (.venv-kt, Python 3.10, needed only for the
           01a/07a MATCH lines and GPU stage-1 training); competition data ~0.8 GB; for
           training, build/ takes about 2.2 GB of shipped files when assembled, plus ~0.6 GB for
           the train transcripts unpacked from the zip (an unpacked train_transcripts/ folder in
           data/raw/, or in the --raw folder, is linked instead), and about 5 GB after the full
           CPU rebuild (our run, which linked an unpacked folder, took 4.4 GB, 3.8 GB on disk)
   ```

   Install uv (Python manager) and the two Python versions:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"     # or open a new shell
   uv python install 3.12.13
   uv python install 3.10.20                # for the MATCH lines of the quick check and the full rebuild
   ```

   The `uv pip install` commands below were checked with uv 0.7.13. Python 3.12.13 and 3.10.20
   need a uv release that lists them (the installer above fetches the current one); any 3.12 /
   3.10 patch release is expected to work, but these are the ones we used.

2. **Python environment.** `requirements.txt` lists the inference packages at the competition
   runtime's 2026-06-15 pins plus xgboost 3.3.0 (its header says how the image that scored the final
   submission differed: pandas 2.3.3, peft 0.18.1, the CUDA 12.9 build of torch);
   `src/training/requirements-cpu.txt` is the freeze of the environment our 2026-10 CPU re-run used
   (see its header) and a superset of `requirements.txt`. One environment serves both:

   ```bash
   uv venv --python 3.12.13 .venv && . .venv/bin/activate
   # the PyTorch cu128 index also hosts packages such as certifi at other versions;
   # --index-strategy lets uv pick the pinned version from either index
   uv pip install --index-strategy unsafe-best-match -r src/training/requirements-cpu.txt
   ```

   For the byte-identical MATCH lines of steps 01a and 07a (quick check and full rebuild), also
   create the Python 3.10 environment and point `KT_PYTHON` at it (07a additionally needs an
   AVX-512 Intel CPU for `MATCH`; see "Byte-identical or value-identical" under Run training):

   ```bash
   uv venv --python 3.10.20 .venv-kt && uv pip install --python .venv-kt -r src/training/requirements-lora-stage1.txt
   [ -x .venv-kt/bin/python ] && export KT_PYTHON=$PWD/.venv-kt/bin/python
   ```

   `.venv-kt` takes about 6 GB and serves only those two MATCH lines and GPU stage-1 training.
   CPU-only users can skip it: without it, 07a prints `DIFFERS` by a few units in the last place
   (see the quick check). Do not set `KT_PYTHON` to a path that does not exist:
   `src/run_training.py` then refuses to start.

3. **Base model of the LLM half** (Apache-2.0, 16 GB), at the exact revision the adapter was
   trained on. The competition runtime mounted it; locally, download it (the `hf` command comes
   with `huggingface-hub==0.36.2` from step 2):

   ```bash
   hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
       --local-dir models/huggingface_models/Qwen/Qwen3-8B
   bash src/tools/check_base_model.sh          # 7 lines ending in ": OK"
   ```

   (`make base-model` runs both commands.) If you keep the model in the Hugging Face cache instead:
   a download pinned to a commit hash writes no `refs/main` entry there, so offline
   loading of `Qwen/Qwen3-8B` fails; `python src/run_inference.py --check` detects this and prints
   the one-line fix.

4. **Competition data** into `data/raw/`, from the data page of the
   [competition](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/) on
   the K-12 AI Infrastructure Platform. You need an account there that has joined the competition
   and accepted its rules. The loaders match these patterns, so DrivenData's random suffixes and a
   browser suffix such as `(1)` do not matter:

   | pattern | served as (example) | needed for |
   |---|---|---|
   | `train_features*.csv` | `train_features_TMQTWsB.csv` | training |
   | `train_labels*.csv` | `train_labels_44ujmj2.csv` | training (also restores the out-of-fold labels) |
   | `train_transcripts*.zip`, or a folder `train_transcripts/` | `train_transcripts.zip` | training |
   | `test_features*.csv` | `test_features_fRc23PD.csv` | inference |
   | `submission_format*.csv` | `submission_format_ZQLcKx7.csv` | inference (row order of the output) |
   | `test_transcripts*.zip`, or a folder `test_transcripts/` | `test_transcripts.zip` | inference |

   - DrivenData serves two format files: `submission_format_ZQLcKx7.csv` (10,508 rows) is the
     format of the released test set; `submission_format_5muR4s3.csv` (100 rows) is the
     platform's smoke test and does not match the released test features.
     `src/run_inference.py` keeps only a format file whose ids are all in the features and takes
     the one with the most rows, so both may sit in `data/raw/`.
   - `test_labels.csv` and `qualitative_metadata.csv` are not needed for inference or training.
     DrivenData released them after the competition, together with the test transcripts; they are
     not in this repository. `test_labels.csv` is used only for optional scoring (below); the
     data audit (`make audit`) also needs it.
   - The released test zip has a stale CRC record for `hkohkub.csv`; the file's contents are
     intact. `src/run_inference.py` reads the zip directly. To unpack it instead:
     `unzip -o -q data/raw/test_transcripts*.zip -d data/raw/test_transcripts || true`, then
     `ls data/raw/test_transcripts/*.csv | wc -l` should print 8241.

5. **Check** (for everyone; nothing is written):

   ```bash
   sha256sum -c --quiet MANIFEST.sha256 && echo REPOSITORY INTACT
   bash src/tools/check_base_model.sh
   python src/run_inference.py --check         # format file, inputs, versions, GPU, base model, asset manifest (add --cpu on a machine without a GPU)
   ```

   `--check` reads the test files in `data/raw/` (or the files given with `--features` and
   `--transcripts`) and ends with `CHECK PASSED`, or `CHECK PASSED WITH <n> WARNING(S)`. It stops
   with an `ERROR` if Qwen3-8B is in none of the places `main.py` looks (step 3). Without the
   competition data, the same preflight runs on the shipped example:
   `python src/run_inference.py --check --cpu --features data/examples/new_data/rows.csv --transcripts data/examples/new_data/transcripts`.

   The training checks (`src/run_training.py --check` / `--verify`) are at the top of
   [Run training](#run-training); they need the `train_*` files and assemble `build/workspace/`.

**Network access.** Setup needs the internet (the uv installer, PyPI, the PyTorch wheel index and
the Hugging Face hub for Qwen3-8B). Inference is fully offline once Qwen3-8B is downloaded. The CPU
rebuild is offline. The GPU retraining
first downloads each model at its pinned revision and writes the `refs/main` entry that offline
loading needs ([Retraining the GPU steps](#retraining-the-gpu-steps)). It then runs offline:
`src/training/reproduce.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the annotation
steps (02), and `run_kt_lsf.sh` sets both for LoRA training (06).

**Shortcuts.** `make help` lists every target: `requirements`, `base-model`, `verify-repo` (the
manifest check, for everyone), `check` and `verify-intermediates` (training checks), `quick-check`,
`rebuild`, `predict`, `example`, `submission`, `audit`, `clean-cache`, `clean`,
and for maintainers:

- `manifest` regenerates `MANIFEST.sha256` and, when the seed replicates are present,
  `models/llm/seeds/MANIFEST.sha256`;
- `manifest-check` checks that the manifests list exactly the shipped files (`verify-repo`
  checks their contents, including the seed replicates when they are unpacked);
- `archive` regenerates the manifests, packs `../trace-the-ace_prize-winner.zip` and
  `../trace-the-ace_prize-winner_seed-replicates.zip` from exactly the files they list (so nothing
  under `build/` can slip in), unpacks both into an empty folder and checks them there, and writes
  the sha256 of each to a `.sha256` file beside it. It needs about 10 GB free next to the output
  and about 5 GB in `$TMPDIR`.

`manifest` and `archive` refuse to run while `data/processed/` holds anything besides `.gitkeep`
(`make predict` and `make example` write there), while `__pycache__` folders exist (run
`make clean-cache`), or while a shipped Markdown file still holds a release placeholder (the
script names the strings it looks for). Run the targets inside the activated `.venv`, from the
repository root.

---

# Hardware

**Inference.** The scored submission ran on the competition platform: 1× NVIDIA A100 80 GB, 24 vCPU,
220 GB RAM. Our post-competition run of the full released test set used 1× NVIDIA H100 PCIe 80 GB on
the DTU HPC cluster (8 cores requested; LSF reported a peak of 11.6 GB). LSF's memory figure is
lower than the maximum resident set: for the 2026-10-19 `src/run_inference.py` run on an H100 it
reported 8.8 GB, while the maximum resident set was 15.5 GiB.

- `src/run_inference.py` needs a GPU with ≥ 40 GB of memory at the shipped batch of 8: the
  worst case is 33.4 GiB (a probe pinned at 8 × 8,192 tokens). The wrapper sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which keeps the reserved memory close to what
  is allocated: on the 100 smoke-test rows (measured on a B200) the default allocator reserved
  64.1 GiB and the setting 32.6 GiB, with bit-identical predictions. The full test set ran on an
  A100 80 GB (platform) and an H100 80 GB. A 100-row smoke test of the final zip also completed on
  an L40S 48 GB (179 s); the full test set was not run
  there through `src/run_inference.py`. The LLM half alone (`src/tools/score_llm_half.py`, the same
  `run_llmkt` code) ran the full test set at batch 8 on an A100 40 GB, an L40S 48 GB and an A40
  48 GB (peak 31.8 GiB), and at batch 2 on 24 GB cards (A10, RTX A5000; peak 19.4 GiB); that tool
  writes no submission. A 16 GB card cannot hold the model, and a V100 (Volta) cannot run the
  shipped torch build.
- The 10,508 test rows took 2 h 01 min on the H100 (120 min 54 s, ≈ 0.69 s per row end to end;
  LSF job 29267305). Through `src/run_inference.py` (2026-10-19) they took 1 h 54 min on an H100
  PCIe and 2 h 38 min on a DTU A100 80 GB PCIe. On the platform's A100 expect ~3–4 h: the
  projection for this zip from its smoke test was 3 h 38 min, inside the 6 h limit. Most of the
  time is the LLM forward pass (~0.8 s per row on an A100); the XGBoost features take ~17–35 min on
  CPU (serial).
- Without a GPU, the one-row example takes ~4 min on the 6-core workstation below (206 s and 236 s
  in two runs, most of it loading the base model and merging the adapter). A full-length
  transcript on CPU has not been timed; we expect minutes per row, so use a GPU for the test set.

**Training** (the runs that produced the shipped files):

| step | hardware | wall clock |
|---|---|---|
| 01 data preparation | CPU | ~30 min |
| 02a answer correctness (Qwen3.5-122B) | UCloud, 8× NVIDIA B200 (4 vLLM shards × 2 GPUs) | ~3 h 35 min |
| 02b NTO tutor moves (Qwen3-32B) | DTU HPC, 3 jobs on 1× H100 PCIe 80 GB each | ~18 h each |
| 02c student states (Qwen3-32B) | DTU HPC, 3 jobs on 1× H100 each | ~2.7 h each |
| 02d interview codebook (Qwen2.5-14B) | UCloud, 1× B200 | ~75 min |
| 02e taxonomy v2 (Qwen3.5-122B) | UCloud, 2× B200 | ~2–2.5 h |
| 02f MRBench correctness (Qwen3-32B) | DTU HPC, 1× H100 | ~13 min |
| 03 distillation (6 classifiers) | DTU HPC CPU nodes, one job per annotator (8–20 cores requested; about one core used per process, per LSF accounting) | 21–41 min per job, including start-up |
| 04 session feature tables | CPU, 8 cores | ~15 min |
| 05 XGBoost (5 seeds) | CPU, 12 cores, 10 GB RAM | 44 min |
| 06a LoRA stage 1 | 1× H100 PCIe 80 GB | 51.5 h |
| 06b LoRA stage 2 (shipped adapter) | 1× B200 | 15 min |
| 06c out-of-fold LoRA models for the calibrator | fold parents: 1× H100 each; MRBench continuation + held-out scoring: fold 1 on 1× H100, folds 2–3 on 1× B200 | 3 × ~41 h (parents, in parallel); fold 1 ~10 min + 1 h 31; folds 2–3 ~15 min + ~35 min each |
| 07–09 calibration, blend report, packaging | CPU | minutes |

About 240 H100-hours and 36 B200-hours in all (~275 GPU-hours; the two A100 fold runs above are
not counted). **None of the GPU work has to be
repeated** to use or rebuild the model: its outputs ship in `data/interim/` and `models/`, and the
CPU rebuild below (3 h 50 min on the workstation) reproduces the submission from them.

**Machine specs**

| machine | used for | specs |
|---|---|---|
| workstation | CPU rebuild, CPU example | Debian 12 (bookworm); Intel Xeon W-2133, 6 cores / 12 threads; 125 GB RAM; Quadro RTX 5000 16 GB + RTX A5000 24 GB, NVIDIA driver 535.309.01 |
| DTU HPC (LSF) GPU nodes | 02b, 02c, 02f, 06a, 06c fold parents, the H100 test run | 1× NVIDIA H100 PCIe 80 GB per job, 4–8 cores; CPU model and OS not recorded in the logs |
| DTU HPC (LSF) CPU nodes | 03, 04, 05 | 8–20 cores requested per job. The job logs record host names, not CPU models; we looked the hosts up with `lshosts` on 2026-10-21, after the fact: the shipped distiller fits (03a, 03c to 03f) ran on n-62-11-63, n-62-11-66, n-62-11-67 and n-62-12-2 (Intel Xeon Gold 6226R) and n-62-13-7 (AMD EPYC 9554; 03d), step 04b on n-62-31-18 (Xeon Gold 6126), and the shipped XGBoost model on n-62-30-23 (Xeon Gold 6142). OS not recorded |
| DTU HPC (LSF) GPU nodes, 2026-10-19 | the GPU-noise runs and the end-to-end `src/run_inference.py` runs | AlmaLinux 9.8, NVIDIA driver 615.71.09; H100 PCIe node: AMD EPYC 9354, 755 GB RAM; A100 80 GB node: Intel Xeon Gold 6326, 1 TB RAM |
| UCloud nodes | 02a, 02d, 02e, 06b, 06c folds 2–3 | NVIDIA B200 (178 GiB = 191.5 GB visible per card; the shipped job logs do not record it). CPU, RAM and OS of the training runs not recorded. The B200 node that scored the seed replicates (all but full seed 222, scored on an L40S) had an AMD EPYC 9655 (`models/predictions/test/seeds/*_b200.json`), and the 2026-10 UCloud jobs ran an Ubuntu 24.04 user space |
| competition platform | the scored runs | 1× NVIDIA A100 80 GB, 24 vCPU, 220 GB RAM |

**GPU compatibility of the training environments.** We trained on 80 GB cards (H100 PCIe, A100
80 GB) and on B200s; the table above is what we used, not a minimum. The stage-1 environment
(torch 2.3.0, CUDA 12.1) has no kernels for Blackwell, so it does not run on a B200. The stage-2
environment (torch 2.11, CUDA 12.8) has none for Volta. The vLLM
environment is built for CUDA 13 and needs NVIDIA driver R580 or newer.

---

# Run training

**Entry point:** `python src/run_training.py`. It lists, runs and checks every step of the pipeline,
in order. Each step prints its command, runs it, and compares what it wrote with the shipped file
(`MATCH` = byte-identical; otherwise a value-by-value comparison). It needs the `train_*` files in
`data/raw/` (or in another folder, see below); the first call that runs or checks anything copies
about 2.2 GB of shipped files into `build/workspace/` and unpacks the train transcripts zip there
(another ~0.6 GB; an unpacked `train_transcripts/` folder is linked instead); `build/` reaches
about 5 GB after the full CPU rebuild.

```bash
python src/run_training.py --list                  # all steps, with run time and what to expect (assembles nothing)
python src/run_training.py --check                 # versions (pyarrow included) and data
python src/run_training.py --verify                # sha256 of every intermediate file: OK / CORRUPT / MISSING,
                                                   # and REBUILT MATCH / REBUILT DIFFERS for files a step rebuilt
[ -x .venv-kt/bin/python ] && export KT_PYTHON=$PWD/.venv-kt/bin/python   # for the MATCH lines below (Setup step 2)
python src/run_training.py --steps 01b,01c,07a,04b,04c,05b,08a,09a,09b   # quick check, ~15 min
python src/run_training.py --from 01 --to 09       # every CPU step from the raw data, ~4 h
```

`--verify` exits non-zero only for MISSING or CORRUPT files; before any step has run it ends with
`61/61 shipped intermediate files verified; 2 to be regenerated from the raw data by steps
01a/01b`. `make check`, `make verify-intermediates`, `make quick-check` and `make rebuild` run the
same commands.

**How it works.** The pipeline runs inside one working directory, `build/workspace/`, laid out the
way it was when we trained (`artifacts/`, `data/`, `external/dialogue-kt/…`). The first call
assembles it with `src/make_workspace.py`:

1. copies the code in `src/training/`;
2. copies every shipped intermediate file from `data/` and `models/` to the place the pipeline
   expects it ([`src/training/workspace_map.tsv`](src/training/workspace_map.tsv) lists each file's
   repository path, workspace path and sha256);
3. links the raw data from `data/raw/` (or the folder given to `make_workspace.py` with `--raw`,
   see below) and unpacks the transcripts;
4. puts the training labels back into the out-of-fold files they were removed from, from
   `train_labels.csv`, and checks each restored file's sha256 against the original.

A full assembly ends by writing `build/workspace/.workspace_ready`, which records the raw-data
folder it used (`raw=…`). Later calls only refresh the code and re-link that recorded folder
(`data/raw/` if none is recorded), so the outputs of steps you re-ran are kept. To train from
competition files kept elsewhere, assemble once with
`python src/make_workspace.py --raw /path/to/competition/files --force`; the `run_training.py`
calls that follow reuse that workspace and keep reading from that folder.
`python src/run_training.py --rebuild-workspace …` starts again from the shipped files (re-linking
the recorded folder): it discards the outputs of steps you re-ran (each one that differed is
printed as `OVERWROTE`) and the shipped copies kept in `build/workspace/artifacts/_bundle/`.

Step paths below (`artifacts/…`, `external/dialogue-kt/…`) are relative to `build/workspace/`; the
"shipped as" entries say where the same file is in this repository. Trained weights are written to
`build/workspace/artifacts/` (LoRA adapters to `build/workspace/external/dialogue-kt/saved_models/`),
and a rebuilt submission ends up in `build/workspace/build/submission/` and
`build/workspace/build/submission.zip`. Every trained weight already ships in `models/`
(~1.7 GB with the zip, plus 2.1 GB of optional seed replicates; inference loads only
`models/submission/`, ~290 MB, plus the 16 GB base model), so nothing has to be retrained to run
inference.

**The quick check** (~15 min; 12 min 16 s measured on the workstation, 2026-10-18) regenerates the
turn table (01b, from the raw transcripts) and the taxonomy-v2 session block (04b, from the shipped
turn probabilities), refits only the LLM calibrator (07a), re-scores the shipped XGBoost model, runs
the train/serve parity gates and repackages the submission. It does not retrain the XGBoost model
(05a) or the distilled classifiers (03a–03f). The lines to compare, abridged (`reproduce.py` prints
a `=== <id> <title>` header per step, followed by indented `MATCH` / `DIFFERS` lines and the
scripts' own output):

```
07a  MATCH     artifacts/llmkt_calibrator_mrft_sub0808.joblib  (sha256 identical to the shipped file)  <- KT_PYTHON, AVX-512 Intel CPU
04b  MATCH     artifacts/v2fam_shrunk_122bC.csv
05b  v2shrunkD   0.6295  64.50 |  0.5718  66.61     <- XGBoost-clean LL, AUROC (16,687 rows) | pooled LL, AUROC (35,072)
08a  LLM alone (calibrated): LL 0.6232 / AUROC 65.53 ... v2shrunkD  w*=0.50 LL 0.6165 AUROC 66.90 | @0.43 0.6166/66.86
09a  46 files byte-identical to the shipped assets/
09b  entry check vs models/submission.zip: identical=49 different=0 missing=0 extra=0
```

- `w*` is the in-sample log-loss-optimal XGBoost weight on the 7,738 clean rows (bootstrap 95 %
  interval [0.39, 0.59]). `@0.43` is the shipped weight; it scores 0.6166 / 66.86 (the
  scorecard's clean column).
- 34 of the 46 assets in 09a are code and configuration copied verbatim from `models/submission/`;
  the line is a packaging check.
- Without `KT_PYTHON`, or on a CPU without AVX-512, 07a prints `DIFFERS` with both maps (a and b
  up to 7 units in the last place apart, ~1e-16 per prediction); 09a then reports 45 of 46 and 09b
  `identical=47 different=2` (`assets/llmkt_calibrator.joblib` and the regenerated
  `assets/manifest_sha256.txt`). This is harmless: step 09b lists differing entries without failing
  (a bad CRC or a missing or extra entry still fails it), and the run continues.

**The full CPU rebuild** (`--from 01 --to 09`) regenerates everything except the GPU outputs from
the raw transcripts. It reproduces our files only on a CPU with AVX-512; on an AVX2-only CPU it
gives a comparable model, not ours (see "Byte-identical or value-identical" below). Of the 46
assets, 8 are retrained by the CPU steps (05a XGBoost, the six distilled
classifiers from 03a–03f, the 07a calibrator); the other 38 are not regenerated: 34 are code and
configuration copied verbatim from `models/submission/`, `nc_curriculum.csv` is a frozen web scrape,
the two LoRA files are shipped GPU outputs, and `lora_sha256.txt` is recomputed from them.

We ran it on the workstation (Machine specs) from 2026-10-17 20:17 to 2026-10-18 00:07, 3 h 50 min
of wall time, with `KT_PYTHON` set, in a fresh `build/workspace/` assembled with `--raw` pointing at
our copy of the competition data, whose `train_transcripts/` folder was already unpacked and was
therefore linked, not extracted; `build/` then took 4.4 GB (3.8 GB on disk), about 0.6 GB less than
when the transcripts are unpacked from the zip. Every file of stages 01, 04 and 07 that is compared
byte for byte came out `MATCH`; the folds and the parity gates passed; 05a's five boosters and its
out-of-fold predictions are identical to ours; and the scores were those of the step tables below
(05b 0.6295 / 64.50 clean and 0.5718 / 66.61 pooled; 08a `w*=0.50` 0.6165 / 66.90, 0.6166 / 66.86 at
the shipped weight 0.43, interval [0.39, 0.59]). Of the distillers, 03a–03e reproduced their classifiers and session
tables byte for byte on that machine (`stu_dst_features.csv`: equal values, within 1.1e-16); their
per-turn probability pickles have equal values and differ in bytes only because pandas stores their
strings differently. 03f's two classifiers differ in bytes and give equal turn probabilities. Steps
03a–03e of that run still used the code from before the vocabulary re-save (see Submitting the model
again, under Run inference), so their four classifiers were compared with the files as they were
before it; re-saving them with sorted vocabularies made them byte-identical to the shipped files.
The training scripts now save in sorted form directly. With the current code, 09a reports 43 of 46
assets byte-identical at best (the XGBoost `baseline.joblib`, `v2t_clf.joblib` and `v2s_clf.joblib`
always differ in bytes and compute identical predictions) and 09b `identical=45 different=4` (those
three and the regenerated `assets/manifest_sha256.txt`), listed without failing. That needs the
conditions of the next paragraph; our clean-room release check of 2026-10-21 (a copy unpacked from
the release archive, environments built from this README, Xeon Gold 6142) met neither of them: 07a
ran with 16 BLAS threads, which the step did not yet pin, and uv had installed a newer Python build.
It measured 38 of 46 and `identical=40 different=9`: the calibrator and the four classifiers of
03a–03e also differed in bytes, with every value equal. The rebuilt XGBoost model's predictions on
the released test set equal the shipped ones on all 10,508 rows (max |Δp| 5e-16).

**Byte-identical or value-identical.** Two retrained outputs reproduce the shipped bytes only under
conditions that a requirements file cannot pin; when the conditions are not met they still compute
the same predictions (to ~1e-16 for the calibrator, exactly for the classifiers), provided the CPU
condition after the list holds:

- *07a, the calibrator* (two floats). Their last bits depend on the numpy build, the CPU and the
  number of BLAS threads, not on the Python version as such. Step 07a runs with one BLAS thread;
  with `KT_PYTHON` (numpy 1.26.4) on an AVX-512 Intel CPU it gives the shipped
  `a=0.6876178756942771 b=0.45650491437726304` (measured on Xeon Gold 6126, 6142 and 6226R with 1
  to 6 threads; 8 or more threads give `…772 / …315`). On an AVX2-only CPU it gives `…772 / …315`
  (1 and 2 units in the last place); in the Python 3.12 environment `…771 / …634` on AVX-512 (b 7
  units away).
- *03a, 03c, 03d, 03e, the four distilled classifiers.* scikit-learn stores in every fitted
  vectorizer the attribute `_stop_words_id = id(self.stop_words)`, here `id(None)`: the address of
  `None` in the Python binary, which in uv's Python builds is a constant of the build. The shipped
  files were written by the python-build-standalone build `20260623` of CPython 3.12.13
  (`python -c 'print(hex(id(None)))'` prints `0x1443f80`); uv 0.11.24 installs build `20260807`
  (`0x1448e50`). With another build the four files differ in that 4-byte field and in nothing else:
  in our release check, writing the shipped value into the field and recompressing gave the shipped
  files byte for byte. The field only tells scikit-learn whether it has already validated the stop
  words; it does not change any prediction. (uv records the build in the `BUILD` file of its Python
  installation.)

**The CPU decides whether the distilled classifiers come out the same at all.** numpy and scipy
bring their own OpenBLAS, which picks its kernels by CPU. Our classifiers were fit where it ran its
`SkylakeX` kernels: the CPUs we fit or reproduced them on (Xeon Gold 6126, 6142 and 6226R, AMD EPYC
9554, the workstation's Xeon W-2133) all have AVX-512, and the three we printed the kernel on (Gold
6126 and 6142, EPYC 9554) used `SkylakeX`. On an AVX2-only CPU OpenBLAS runs its `Haswell` kernels,
which round differently, and the logistic regressions of 03a–03f end at different weights from the
same vocabularies. To see which kernel you have:

```bash
python -c "import threadpoolctl; print({i['architecture'] for i in threadpoolctl.threadpool_info() if i['user_api'] == 'blas'})"
```

Our release check of 2026-10-27 ran the full CPU rebuild on a Xeon E5-2660 v3 (AVX2 only). The
largest change in a single turn's probability was 0.15 to 0.97, depending on the classifier, and
everything downstream changed with them: 03a `p_ans` OOF AUC 0.8247 (ours 0.8253), 03c 0.8185
(0.8219), 05b 0.6295 / 64.53 on the clean rows and 0.5717 / 66.65 pooled (0.6295 / 64.50 and
0.5718 / 66.61), 08a `w*=0.50` 0.6165 / 66.93 (0.6165 / 66.90). That is a comparable model, not
ours. Refitting 03e on a Xeon Gold 6126 gave our weights exactly, and with
`OPENBLAS_CORETYPE=Haswell` the AVX2 weights exactly, so the kernel alone accounts for the
difference. We saw no effect on inference: on the same AVX2 CPU the quick check matched every line
and the one-row example gave our probability exactly.

**The GPU steps** (02 annotation, 06 LoRA) are skipped unless you pass `--with-gpu`; their outputs
ship in the repository, and re-running them costs ~275 GPU-hours and is not bit-reproducible. They
need a GPU cluster (8 GPUs for step 02a, the vLLM environment, two LoRA environments, the Qwen
models in the Hugging Face cache). `--with-gpu` runs the GPU commands but cannot run the whole
pipeline unattended: it stops after 02a (merge the annotation shards) and after 06c (pool the
out-of-fold predictions), and prints how to continue. Before any GPU step runs, `--with-gpu` checks that
`VLLM_PYTHON`, `KT_PYTHON` and `KT2_PYTHON` are set and have their packages (vllm, plus pyarrow
for 02d; torch, transformers, peft, pykt, pyBKT), and that every Qwen model the selected steps need is in
the cache at its pinned revision, with `refs/main` written. The annotation steps (02) run with
`HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and, unless you set it, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
`--from`/`--to` also take a step id, e.g. `--from 02b`. The commands, environments and models are
below.

## Retraining the GPU steps

**You do not need this to use, inspect or submit the model.** It costs about 275 GPU-hours, and the
result is a comparable model, not the same bytes. Every output of these steps already ships
(`data/interim/`, `data/external/mrbench/derived/`, `models/llm/`), so any single step can be re-run
and compared with ours. Run Setup steps 1 to 5 first. The environment and model commands run from
the repository root; the annotation and LoRA commands run inside `build/workspace/`.

**Environments.** Three environments live at the repository root, next to `.venv` (their names must
start with `.venv`, which the release manifest skips):

| env | variable | used by | Python | install |
|---|---|---|---|---|
| LoRA stage 1 | `KT_PYTHON` | 06a, the fold parents of 06c, and the CPU steps 01a and 07a | 3.10 | the `.venv-kt` of Setup step 2 |
| vLLM | `VLLM_PYTHON` | 02 (annotation) | 3.10 | `uv venv --seed --python 3.10.20 .venv-vllm && .venv-vllm/bin/pip install -r src/training/requirements-vllm.txt` |
| LoRA stage 2 | `KT2_PYTHON` | 06b, the MRBench continuations and held-out scoring of 06c | 3.12 | commands below |

```bash
uv venv --seed --python 3.12 .venv-stage2
.venv-stage2/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-stage2/bin/pip install -r src/training/requirements-lora-stage2.txt
.venv-stage2/bin/pip install --no-deps pyBKT==1.4.1 pykt-toolkit==0.0.38   # imported by dialogue-kt
```

`sentencepiece`, `scipy`, `tqdm` and `matplotlib` were installed unpinned in the original stage-2
environment, so their exact versions are unknown; none of them enters the LoRA weight updates.
The stage-1 environment (torch 2.3.0+cu121) cannot train on Blackwell cards such as the B200; the
stage-2 environment (torch 2.11.0+cu128) cannot run on Volta cards such as the V100; the vLLM
environment uses the CUDA 13.0 runtime and needs NVIDIA driver R580 or newer. The 122B FP8
annotator needs ~130 GB of GPU memory (two 80 GB cards with `ANNO_TP=2`); the other annotators and
LoRA training at 8,192 tokens each fit on one 80 GB card.

**Model weights.** Every step names its model by a bare id such as `Qwen/Qwen3-8B`, and an offline
load resolves that id through the file `refs/main` in the Hugging Face cache. A download pinned to a
commit hash does not write that file, so the `pin` helper below downloads each model at the
revision we ran and then writes `refs/main` itself:

```bash
export KT_PYTHON=$PWD/.venv-kt/bin/python          # stage 1 (also 01a, 07a)
export VLLM_PYTHON=$PWD/.venv-vllm/bin/python      # annotation
export KT2_PYTHON=$PWD/.venv-stage2/bin/python     # stage 2
export HF_HOME=/big/disk/hf-cache   # an absolute path; the four models take ~240 GB
pin(){ .venv-kt/bin/hf download "$1" --revision "$2" && d="$HF_HOME/hub/models--${1//\//--}/refs" && mkdir -p "$d" && printf %s "$2" > "$d/main"; }
pin Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218                 # LoRA base (06*)
pin Qwen/Qwen3-32B 9216db5781bf21249d130ec9da846c4624c16137                # 02b, 02c, 02f
pin Qwen/Qwen3.5-122B-A10B-FP8 a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9    # 02a, 02e
pin Qwen/Qwen2.5-14B-Instruct cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8     # 02d
python src/run_training.py --check --with-gpu --steps 02a,02b,02c,02d,02e,02f,06a,06b,06c   # no GPU needed
```

If you set `HF_HUB_CACHE`, write `refs/main` under that folder instead of `$HF_HOME/hub`. Do not
pass a snapshot path as `--base_model` to work around a missing `refs/main`: PEFT writes that value
into the adapter's `adapter_config.json`. When you run an annotator by hand, export
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn` first (`reproduce.py`
sets them for the 02 steps; `run_kt_lsf.sh` sets the first two for LoRA training). Outside a batch
scheduler, set `CUDA_VISIBLE_DEVICES` yourself.

**Annotation (02).** Each step runs through `python src/run_training.py --steps 02b --with-gpu` (and
so on), or by hand as below, from `build/workspace/`. `src/make_workspace.py` copies every shipped
annotation file to the path its annotator writes, and the annotators resume from existing output,
so move a step's shipped output aside before running it by hand (`reproduce.py` does this itself):

```bash
cd build/workspace
python reproduce.py --steps 01a,01b      # LoRA input and distiller input, with KT_PYTHON exported
aside () { for f in "$@"; do mkdir -p "artifacts/_bundle/$(dirname "$f")" && mv -n "$f" "artifacts/_bundle/$f"; done; }

# 02a answer correctness, Qwen3.5-122B (we used 8x B200, ~3 h 35 min): 4 shards, one per pair of GPUs
aside external/dialogue-kt/results/traceace_turn_labels_qwen35_122b.csv
cd external/dialogue-kt
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$((2*i)),$((2*i+1)) ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 \
  ANNO_MAXTOK=4096 ANNO_CHUNK_TURNS=10 ANNO_REF=results/NONEXISTENT.csv ANNO_SHARD=$i/4 \
  $VLLM_PYTHON annotate_traceace_vllm.py > results/anno_prod_shard$i.log 2>&1 &
done; wait
cd ../..

# 02b NTO tutor moves, Qwen3-32B (1x H100 per shard, ~18 h each; optional, the booster drops these columns)
aside external/dialogue-kt/results/nto_moves_shard{0,1,2}.csv
(cd external/dialogue-kt && for s in 0 1 2; do ANNO_SHARD=$s/3 $VLLM_PYTHON annotate_nto_moves.py; done)

# 02c student states, Qwen3-32B (1x H100 per shard, ~2.7 h each)
aside external/dialogue-kt/results/student_states_shard{0,1,2}.csv
(cd external/dialogue-kt && for s in 0 1 2; do ANNO_SHARD=$s/3 ANNO_N_SESSIONS=1600 $VLLM_PYTHON annotate_student_states.py; done)

# 02d interview codebook, Qwen2.5-14B (1x B200, ~75 min)
aside artifacts/doc29_records_14b/doc29_raw_records_14b.parquet
PYTHONPATH=src $VLLM_PYTHON scripts/annotate_doc29_labels.py --qwen Qwen/Qwen2.5-14B-Instruct \
  --transcripts data/train_transcripts --out-dir artifacts/doc29_records_14b --sessions 4000

# 02e taxonomy-v2 tutor and student moves, Qwen3.5-122B (2x B200, ~2-2.5 h)
aside artifacts/v2labels/122b/moves_v2_ext_tutor_shard0.csv artifacts/v2labels/122b/moves_v2_ext_student_shard0.csv
ANNO_MODEL=Qwen/Qwen3.5-122B-A10B-FP8 ANNO_TP=2 ANNO_ARM=ext ANNO_ROLES=tutor,student \
ANNO_SHARD=0/1 ANNO_N_SESSIONS=3072 ANNO_OUTDIR=artifacts/v2labels/122b \
$VLLM_PYTHON scripts/annotate_v2_moves.py

# 02f MRBench correctness labels, Qwen3-32B (1x H100, ~13 min); skip it to stay closest to the shipped adapter
aside external/dialogue-kt/results/mrbench_turn_labels_qwen32b.csv \
      external/dialogue-kt/data/annotated/mrbench_train.pkl external/dialogue-kt/data/annotated/mrbench_test.pkl
(cd external/dialogue-kt && $VLLM_PYTHON annotate_mrbench_vllm.py && $KT_PYTHON build_mrbench_pkl.py)
```

- **02a** writes `results/traceace_turn_labels_vllm_shard{0..3}.csv`. Concatenate the four shards
  and drop duplicate `(session_id, turn)` rows into `results/traceace_turn_labels_qwen35_122b.csv`.
  On fewer GPUs, run the four `ANNO_SHARD=i/4` commands one after another. In our run 13.6 % of the
  chunks failed to parse and were re-run during the merge by a script that was not kept, so a merge
  by hand is approximate; the shipped merged file has 2,038,332 rows over 22,803 sessions.
- **02b** can be skipped with its distiller 03c (about 54 H100-hours): keep the shipped `nto_*`
  outputs, which 05a and inference still read.
- **02e** is reconstructed (the original launcher was not kept); the session list it produces must
  equal `artifacts/v2_sessions_3072.txt`.
- **02f**: re-annotating MRBench shifts the LLM probabilities up slightly, which the frozen
  calibrator does not absorb; to stay closest to the shipped adapter, keep the shipped
  `mrbench_train.pkl` and go straight to 06b.

**LoRA fine-tuning of Qwen3-8B (06),** from `build/workspace/external/dialogue-kt`. Every run: LoRA
`r=16`, `alpha=16`, dropout 0.05, all seven projections; AdamW, weight decay 0.01, gradient clip
1.0, batch 1 x gradient accumulation 64, constant learning rate, bf16, gradient checkpointing, seed
221. Training overwrites the workspace copy of the adapter of the same name; the shipped originals
stay in `models/llm/`. `run_kt_lsf.sh` picks its interpreter only from `KT_PYTHON`, so the stage-2
commands set `KT_PYTHON=$KT2_PYTHON` on the command line.

```bash
cd external/dialogue-kt

# 06a stage 1: all competition dialogues, 2 epochs (1x H100 80 GB, 51.5 h)
TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 \
./run_kt_lsf.sh train --dataset traceace --model_type lmkt --train_all \
  --pack_kcs 0 --quantize 0 --inc_first_label --epochs 2 \
  --base_model Qwen/Qwen3-8B --model_name lmkt_traceace_qwen3_8b_full_final

# 06b stage 2: MRBench continuation, 17 steps (1x B200 or H100, ~15 min) -> the shipped adapter
TRACEACE_PKL=data/annotated/mrbench_train.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON=$KT2_PYTHON \
./run_kt_lsf.sh train --dataset traceace --model_type lmkt --train_all \
  --pt_model_name lmkt_traceace_qwen3_8b_full_final \
  --pack_kcs 0 --quantize 0 --inc_first_label --epochs 1 --lr 1e-4 \
  --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_all35k

# 06c out-of-fold twins for the calibrator, folds 1-3 (fold parents ~41 h each on 1x H100)
for k in 1 2 3; do
  TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 ./run_kt_lsf.sh train \
    --dataset traceace --model_type lmkt --fold $k --pack_kcs 0 --quantize 0 --inc_first_label \
    --epochs 2 --base_model Qwen/Qwen3-8B --model_name lmkt_traceace_qwen3_8b_full
  TRACEACE_PKL=data/annotated/mrbench_train.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON=$KT2_PYTHON ./run_kt_lsf.sh train \
    --dataset traceace --model_type lmkt --train_all --pt_model_name lmkt_traceace_qwen3_8b_full_$k \
    --pack_kcs 0 --quantize 0 --inc_first_label --epochs 1 --lr 1e-4 \
    --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f$k
  ln -sfn lmkt_mrft_sa_8b_a1f$k saved_models/lmkt_mrft_sa_8b_a1f${k}_$k   # test --fold k loads <name>_k
  TRACEACE_PKL=data/annotated/traceace_full.pkl LLMKT_MAX_SEQ_LEN=8192 KT_PYTHON=$KT2_PYTHON ./run_kt_lsf.sh test \
    --dataset traceace --model_type lmkt --fold $k --batch_size 8 --pack_kcs 0 --quantize 0 \
    --inc_first_label --base_model Qwen/Qwen3-8B --model_name lmkt_mrft_sa_8b_a1f$k
done

# pool the held-out predictions into artifacts/mrft_oof_3fold.csv (fold, y, p; last numeric Prob per dialogue)
cd ../..
aside artifacts/mrft_oof_3fold.csv
python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
import pandas as pd
from build_mrft_oof import QUAL, terminal_preds
pd.concat([terminal_preds(QUAL[k]).assign(fold=k)[["fold", "y", "p"]] for k in (1, 2, 3)]
          ).to_csv("artifacts/mrft_oof_3fold.csv", index=False)
EOF
python reproduce.py --steps 07b,07a
cd ../..   # back to the repository root
```

Ours: 06a (job 28869266) `Number of data points: 35062`, epoch losses 0.6430 and 0.5323; 06b 1,032
data points, 17 optimiser steps, epoch loss 0.5607; the pooled file has 7,273 + 4,371 + 5,049 =
16,693 rows. Step 07a refuses to run while the new held-out files are newer than
`artifacts/mrft_oof_3fold.csv`, so it cannot fit on the old adapters' predictions. Keep a copy of
your pooled file: `python src/run_training.py --rebuild-workspace` restores the shipped one.

**Then the CPU pipeline:**

```bash
python src/run_training.py --steps 03a,03b,03c,03d,03e,03f,04a,04b,04c,05a,05b,07a,08a,09a,09b
```

With your own annotations and adapters, 09a reports which assets differ from ours (all retrained
ones will) and 09b writes `build/workspace/build/submission.zip`, listing the differing entries
without failing. Before using a retrained model: the three parity gates in 04c must pass; 05b and
08a should land near our numbers (XGBoost-clean log loss about 0.63, blend about 0.617); refit the
LLM calibrator (07a) on your own out-of-fold predictions, since the shipped Platt map belongs to the
shipped adapter; keep `blend_weight.txt` at 0.43 unless 08a puts the optimum far outside [0.39,
0.59]; and check that 09a lists `DIFFERENT  assets/lora_llm/adapter_model.safetensors` (it stages
the adapter from `saved_models/lmkt_mrft_sa_8b_all35k/`). Then run it with `python
src/run_inference.py --submission-dir build/workspace/build/submission`.

## The steps, with their outputs and the scores you should get

### Stage 01: prepare the data (CPU, ~30 min)

| field | value |
|---|---|
| run | `python src/run_training.py --steps 01a,01b,01c` (01a runs under `KT_PYTHON`) |
| out (01a) | `external/dialogue-kt/data/annotated/traceace_full.pkl`: one row per response: transcript turns + objective + label, the LoRA training table (**not shipped: competition text**) |
| out (01b) | `artifacts/turn_text.pkl.gz`: 2,368,779 student turns with the tutor turns around them, the input of every distiller (**not shipped: competition text**); `artifacts/evidence_features.csv`: contingency and repair features (shipped as `data/interim/session_features/`) |
| 01c | verifies the frozen folds `data/interim/folds/{folds_gkf5_lo,clean_rows_gkf5}.npy` (sha `59f0c807dad3ac08` / `cc365b58eaea4bdb`). They ship frozen and are never regenerated, because GroupKFold's tie-breaking differs between CPUs |
| expect | `merged responses: 35072  unique LO: 398  sessions: 22821`; 35,062 responses written (the 10 responses of 5 sessions (akplwos, mcpjrkq, jzifuvp, jhaptfk, nxjsmda) have tutor and background turns but no student turn; logged as `skipped 10 missing/empty transcripts`); `traceace_full.pkl` byte-identical to ours when written by the Python 3.10 environment |

### Stage 02: annotate turns with large LLMs (GPU, vLLM; outputs shipped)

Each annotator labels every turn of a transcript against a codebook (greedy decoding, Qwen3
thinking off). NTO moves, student states, the interview codebook and taxonomy v2 use guided JSON
output; the two correctness steps (02a, 02f) parse free-form JSON. No annotator retries a reply
that fails to parse: failures are counted and logged. In production 02a, 33,437 of 246,645 chunks
failed to parse (the four shard logs); they were re-run when the shards were merged, by a script
that was not kept. Production 02f had no
failures. The outputs hold session ids, turn numbers and labels only.

| step | annotator | revision | codebook | sessions | shipped as (`data/…`) |
|---|---|---|---|---|---|
| 02a | Qwen3.5-122B-A10B-FP8 | `a099dee7` | answer correctness (the LLMKT prompt) | 22,803 of 22,821 | `interim/annotations/traceace_turn_labels_qwen35_122b.csv` |
| 02b | Qwen3-32B | `9216db57` | NTO tutor moves | 22,811 of 22,821 | `interim/annotations/nto_moves_shard{0,1,2}.csv` |
| 02c | Qwen3-32B | `9216db57` | 12 student states | 4,800 | `interim/annotations/student_states_shard{0,1,2}.csv` |
| 02d | Qwen2.5-14B-Instruct | `cf98f3b3` | interview codebook | 4,000 | `interim/annotations/doc29_raw_records_14b.parquet` |
| 02e | Qwen3.5-122B-A10B-FP8 | `a099dee7` (assumed: the 02e log was not kept) | taxonomy-v2 tutor + student moves | 3,072 | `interim/annotations/moves_v2_ext_{tutor,student}_shard0.csv` |
| 02f | Qwen3-32B | `9216db57` | answer correctness on MRBench | 504 dialogues (404 train stage 2, 100 held out) | `external/mrbench/derived/` |

The revisions are the model snapshots our job logs record; the full hashes are in `BASE_MODELS`
in `src/training/reproduce.py`. Extra annotations: student states on the full
corpus (`data/interim/annotations/extra/`). Commands and hardware:
[Retraining the GPU steps](#retraining-the-gpu-steps).

### Stage 03: distil each annotator into a small classifier (CPU, single-threaded, ~2–4 h)

TF-IDF (word 1–2-grams + character n-grams) → one logistic regression per label, trained on the turn
table against the annotator's labels. Five-fold session-disjoint cross-fitting gives every annotated
turn an out-of-fold probability (what the XGBoost model trains on); a final fit on all labelled
turns is what ships. Classifiers are shipped in `models/distilled/`, turn probabilities in
`data/interim/turn_probabilities/`, session tables in `data/interim/session_features/`.

| field | value |
|---|---|
| run | `python src/run_training.py --steps 03a,03b,03c,03d,03e,03f` |
| 03a | `turn_correctness_clf.joblib`, `turn_correctness_probs.pkl.gz`, `evidence_features_full.csv` (+ 14 answer-log columns) → expect `p_ans: OOF AUC 0.8253`, `p_cor: OOF AUC 0.7304` |
| 03b | `nto_rate_features.csv` (session-level NTO rates, a diagnostic input of 03c) → `MATCH` |
| 03c | `nto_moves_clf.joblib`, `nto_turn_probs.pkl.gz`, `nto_dst_features.csv` → `MEAN OOF AUC 0.8219` (19 heads) |
| 03d | `stu_states_clf.joblib`, `stu_turn_probs.pkl.gz`, `stu_dst_features.csv` → 0.8486 = mean of the 12 printed `AUC` lines (no summary line) |
| 03e | `doc29_clf.joblib`, `doc29_turn_probs.pkl.gz` → `Mean OOF AUC 0.9080 over 14 heads` (in `figures/data/doc29_distil_validation.md`; the console prints the per-head AUCs) |
| 03f | `v2t_clf_v2ext_122bC.joblib`, `v2s_clf_v2stu_122bC.joblib`, their turn probabilities, and `v2{t,s}_features_*.meta.json` (`models/distilled/`) → `0.8845 over 32 heads` / `0.8709 over 12 heads` |

The distillers print AUROC as a fraction: `OOF AUC 0.8253` is the 82.53 of the table under
Summary. They take 13–61 min each on a 6- to 12-core machine; 03b takes a minute. These steps run
single-threaded on purpose: with a multi-threaded BLAS the order of floating-point additions
changes and some cross-fitting folds settle on slightly different (equally good) solutions;
single-threaded, on the AVX-512 CPUs we used, they reproduce our classifiers byte for byte (on an
AVX2-only CPU they do not; see "Byte-identical or value-identical" under Run training).

### Stage 04: session feature tables (CPU, ~15 min)

| field | value |
|---|---|
| run | `python src/run_training.py --steps 04a,04b,04c` |
| out | `doc29_features_distilled.csv` (34 `d29dst_` columns) · `v2fam_shrunk_122bC.csv` (46 `v2ht_`/`v2hs_` columns, rates shrunk towards a frozen corpus mean for short sessions with pseudo-count κ = 25), both in `data/interim/session_features/` |
| expect | both byte-identical; three parity gates (the inference code recomputes the training tables) print `parity OK`, `PARITY PASS — all 12 columns…`, `PARITY PASS — all 34 columns…`. The gates cover the `d29dst_`, `studst_` and `v2ht_`/`v2hs_` blocks, 92 of the 177 XGBoost columns. The other 85 (54 handcrafted, 14 `recdst_`, 10 `ctg_`, 7 `rep_`) have no numeric gate; the last three blocks are built in training by separate code |

### Stage 05: the XGBoost half (CPU, ~45 min, 10 GB)

| field | value |
|---|---|
| run | `python src/run_training.py --steps 05a,05b` |
| out | `artifacts/baseline_v2shrunkD.joblib` (`models/gbm/` = `models/submission/assets/baseline.joblib`) · `artifacts/v2shrunkD_oof.csv` (`models/predictions/oof/`) |
| expect | `DROP: -21 cols matching 'ntodst_' -> 177 numeric`; `ensemble OOF … Platt LL=0.5718 AUC=66.61`; `frozen Platt: a=0.9411 b=-0.0117`; 05b on the 16,687 XGBoost-clean rows: **0.6295 / 64.50**. Released test set: **0.6015 / 62.55** |

Settings: seeds 42, 143, 244, 345, 446; `n_estimators=300, learning_rate=0.05, lossguide,
max_leaves=31, min_child_weight=126, reg_lambda=80, reg_alpha=10, colsample_bytree=0.35,
subsample=0.6, gamma=0.02`; objective TF-IDF (1,457 terms) and objective one-hot (398).

### Stage 06: the LLM half, a two-stage LoRA fine-tune of Qwen3-8B (GPU; outputs shipped)

LoRA r=16, α=16, dropout 0.05 on all seven projections; AdamW, batch 1 × gradient accumulation 64,
constant learning rate, bf16.

| step | what | out (shipped as `models/llm/…`) | ours (seed 221), and our re-runs |
|---|---|---|---|
| 06a | **stage 1**: 35,062 competition dialogues, 2 epochs, lr 2e-4 (1×H100, 51.5 h) | `lmkt_traceace_qwen3_8b_full_final/` | job 28869266: train loss 0.6430 → 0.5323 |
| 06b | **stage 2**: 1 epoch on 1,032 MRBench turns from 404 dialogues, lr 1e-4, from stage 1 (17 steps, 15 min) | `lmkt_mrft_sa_8b_all35k/` = `models/submission/assets/lora_llm/` | first step 0.666, epoch loss 0.5607; scorecard row "LLM half" |
| 06c | out-of-fold models: stage 1 + stage 2 on 3 objective-disjoint folds, scored on the held-out folds | `lmkt_traceace_qwen3_8b_full_{1,2,3}/`, `lmkt_mrft_sa_8b_a1f1/`; pooled predictions in `models/predictions/oof/mrft_oof_3fold_keyed.csv` | the scorecard's out-of-fold column |

Annotation (02, vLLM continuous batching) and LoRA training (06, bf16 and nondeterministic CUDA
kernels) are not bit-reproducible, even with the same seed on the same GPU model; inference at a
fixed batch on the same GPU model is bit-identical.
The values above are therefore ours, not exact targets. How far a retrain with a new seed lands is documented in the seed replicates archive: retraining stage 2 only moved the ensemble
log loss by −0.0005 to +0.0008, retraining both stages by +0.0025 to +0.0111 through the shipped
calibrator. To stay closest to the shipped adapter, skip 02f and train 06b on the shipped
`mrbench_train.pkl`.

### Stage 07: calibrate the LLM half (CPU, seconds)

| field | value |
|---|---|
| run | `python src/run_training.py --steps 07a` (with `KT_PYTHON` set for byte equality) |
| out | `artifacts/llmkt_calibrator_mrft_sub0808.joblib` (`models/calibrators/` = `models/submission/assets/llmkt_calibrator.joblib`) |
| expect | `unconstrained OOF Platt: a=0.6876`, OOF log loss 0.5826 → **0.5653**, `MATCH` |

Recipe: Platt slope `a = min(fitted slope, 0.85)`, intercept solved so that the mean calibrated
prediction equals the training base rate 0.7025. No test quantity enters. Step 07b
(`scripts/build_mrft_oof.py`, keys the pooled out-of-fold predictions by response id) runs only
after re-running 06c; otherwise the shipped keyed file is used.

### Stage 08: blend (CPU, ~5 min)

`python src/run_training.py --steps 08a` → on the 7,738 clean rows: `LLM alone (calibrated): LL
0.6232 / AUROC 65.53`; blend optimum `w*=0.50 LL 0.6165`, bootstrap 95 % interval of the optimum [0.39, 0.59], and
`@0.43 0.6166/66.86` at the shipped weight. The weight 0.43 is frozen and is not refitted.

### Stage 09: assemble and pack the submission (CPU, ~1 min)

`python src/run_training.py --steps 09a,09b` → `build/workspace/build/submission/` and
`submission.zip`; from the shipped intermediate files (quick check, with `KT_PYTHON`): `46 files
byte-identical to the shipped assets/`, `entries: 53  (models/submission.zip: 53)` and `entry check
vs models/submission.zip: identical=49 different=0 missing=0 extra=0` (47 and 2 when 07a printed
`DIFFERS`). In the pipeline, 09b lists differing entries without failing (after a full rebuild: 45
identical and 4 different at best, 40 and 9 in our clean-room check; see above); a
direct call of `src/tools/package_submission.sh` (`make submission`) exits 1 on any difference.

To run the rebuilt model: `python src/run_inference.py --submission-dir build/workspace/build/submission`
(or upload `build/workspace/build/submission.zip`).

**What cannot be reproduced exactly:** the GPU steps (not bit-reproducible even with the same seed
on the same GPU model, see Stage 06); and gaps in our records: the 02a shard-merge script and the
02d and 02e launchers were not kept; the 06c pooling script was not kept and two fold adapters were
overwritten; and stage-1 LoRA training ran from an uncommitted working tree of our dialogue-kt fork,
on a different transformers/peft stack from stage 2. Their outputs all ship. The launchers we kept
are in `src/training/launchers/original/`.

---

# Run inference

**Entry point:** `python src/run_inference.py`. It runs `models/submission/main.py`, the exact
program the competition platform ran, unchanged, around your input files. No retraining or manual
input.

```bash
# the competition test set in data/raw/  ->  data/processed/submission.csv
python src/run_inference.py
#   format: submission_format_ZQLcKx7.csv (10508 rows) ... PROBE manifest OK 46 files verified ...
#   MODEL_INTEGRITY=OK ... PROBE blend w=0.43 missing=0 ... inference complete
#   wrote .../data/processed/submission.csv (10508 rows)

# check the result: row-by-row against our H100 run, and (optional) the score
python src/tools/score_submission.py data/processed/submission.csv                                    # vs our H100 run
python src/tools/score_submission.py data/processed/submission.csv --labels /path/to/test_labels.csv  # also log loss / AUROC

# any new data: a CSV of rows + a folder (or zip) of transcripts (add --cpu on a machine without a GPU)
python src/run_inference.py --features data/examples/new_data/rows.csv \
    --transcripts data/examples/new_data/transcripts --out data/processed/example_predictions.csv
#   r1,0.649216874889283 on CPU (data/examples/new_data/expected_predictions_cpu.csv)
```

`make example` runs the last command with `--cpu`, then compares the result with
`data/examples/new_data/expected_predictions_cpu.csv` through
`src/tools/score_submission.py --max-diff 0.02`, and fails (exit 1) if the prediction differs by
more than 0.02. Delete `data/processed/example_predictions.csv` afterwards if you are a maintainer
about to run `make manifest` or `make archive`.

Options: `--features`, `--transcripts` and `--out` as above; `--submission-format` (which rows to
write, in which order; by default the right file is picked from `data/raw/`, or the order of
`--features`); `--check` (preflight only: format selection, inputs, package versions, device, base
model, asset manifest; loads no model; ends with `CHECK PASSED` or
`CHECK PASSED WITH <n> WARNING(S)`); `--cpu` (allow running without a GPU, for a handful of rows);
`--submission-dir` (default `models/submission`; e.g. a rebuilt tree); `--skip-version-check`;
`--allow-env-override` (keep your own `LLMKT_MAX_SEQ_LEN` / `LLMKT_KEEP_HEADER`, which may make the
predictions differ from the scored run; without it the wrapper sets the scored values, 8192 and 1).
The wrapper sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` unless you set it yourself,
warns if the `flash_attn` package is installed (it is not in the runtime's lock file at
commit ea9a817),
and
prints the number of rows written.

In the `PROBE` lines, `llm_scored` and `missing` are counted after rows without a usable
transcript have been filled with the fallback, so `missing=0` does not show that every transcript
was found; the wrapper's `WARNING: … session(s) have no usable transcript` line (and its
`transcripts: N of M session(s) found` line) is the check for that.

**Expected result.** Per-row agreement with `p_blend` in
`models/predictions/test/shipped_components.csv` (our H100 run) to about ±0.01 (at most ~0.025) on
another GPU or on CPU (the XGBoost half is exact), and log loss 0.5934 ± 0.0002 / AUROC 64.7 on the
10,508 released rows. Our reference is a run of the shipped `main.py`, called directly on the H100
(LSF job 29267305), with one inert, environment-gated block added that wrote the two halves to a
side file (`COMPONENT_DUMP`); that side file is `shipped_components.csv`. It scored 0.5934 / AUROC
64.73. It used the files as scored, before the vocabulary re-save and comment edits described below,
which leave every transform identical. `src/run_inference.py` on the full test set with the shipped
(re-saved) files reproduces it bit for bit on an H100 PCIe (0.59345 / 64.73, all 10,508
probabilities identical to `shipped_components.csv`) and to within GPU arithmetic on an A100 80 GB
(0.59343 / 64.74).

Inputs: rows with `response_id, session_id, learning_objective_id, learning_objective`, and one
transcript per session with `session_id, utterance_id, role, content, timestamp` (role
`tutor`/`student`/`background`; timestamp `HH:MM:SS` elapsed; `utterance_id` a whole number,
unique within the transcript: the wrapper warns, without stopping, when an id or a timestamp does
not fit). An objective never seen in training is the expected case: an unseen
`learning_objective_id` gets no adjustment, and the model uses the objective's text. A reused
training id shifts the XGBoost half slightly (only 6 of the 398 training ids are ever split on). Use
one id per objective text, because the curriculum feature `lo_ks_level` is computed once per id.
Each row is scored independently. 75 test sessions (92 rows, 0.9 %) reuse
byte-identical training transcripts, always under a different objective; those transcripts were
seen in training, and the train-keyed shipped tables include them.

What happens, in order (all inside `main.py`):

1. every asset is re-hashed against `models/submission/assets/manifest_sha256.txt`;
2. **LLM half**: Qwen3-8B is loaded in bf16 on CPU, the LoRA adapter is merged (on CPU: peft
   upcasts the `B @ A` product to fp32 only there, so a GPU merge gives different weights), the
   model moves to the GPU; one prompt per row, scored in length-sorted batches
   of 8; the frozen Platt map is applied;
3. **XGBoost half**: handcrafted features, then the six distilled classifiers (from five codebooks)
   over every turn, aggregated into session features; the NTO pass still runs, but its 21
   `ntodst_` columns were removed before training (`--drop-cols ntodst_`) and are ignored, kept so
   the scored code stays unchanged; then the 5-seed XGBoost model with its own Platt map;
4. `p = 0.57 · p_LLM + 0.43 · p_GBM`, clipped to [1e-6, 1 − 1e-6] → `submission.csv`.

Space: the transcripts of the requested sessions are copied to a temporary directory under
`$TMPDIR` (~190 MB for the test set; set `TMPDIR=/some/dir` to move it; deleted at exit). The output is one
CSV.

**Reproducibility across machines.** The XGBoost half reproduces exactly on any machine. The LLM
half's True/False logits are bf16; on a different GPU, or with a different batch size, a logit can
round to the neighbouring value, moving one row's final probability by about 0.01 (at most 0.024
measured). A repeated run on the same GPU model with the same batch is bit-identical, even on
another node. Across eight full-test runs on six GPU models (H100, A100 80/40 GB, L40S, A40, A10,
RTX A5000) the score varies with sd 0.00004 in log loss and 0.012 AUROC points.

**Retraining with another seed.** We retrained the LLM half's LoRA adapters (not the calibrator)
with four new seeds (adapters in `models/llm/seeds/`, from the seed-replicates archive). These runs
estimate variance under a changed seed, and for the full retrains also a changed GPU and
environment; they are not reproductions.
Retraining only stage 2 moves the ensemble log loss by −0.0005 to +0.0008. Retraining both stages
varies more: three of the four seeds land within 0.003 of the shipped adapter when both are
compared uncalibrated, and 0.0025–0.0065 worse through the shipped Platt map, which was fitted for
the shipped adapter's lineage; the fourth seed diverged early in training. A faithful retraining
refits the calibrator with the same seed (steps 06c and 07a).

**Submitting the model again.** `models/submission.zip` (sha256
`e2a8d2f28eaf3274a75413a13256feb42129ca311a538a2854078cb11dc8d2f6`) is the scored submission with
its seven text-classifier files re-saved so that their vocabularies are stored in sorted order,
and comments and docstrings edited in 19 source files (`main.py`, `main_llmkt_lora.py` and 17
modules in `assets/magnificat/`; no code changed); predictions are identical. Those 26 files and
the regenerated `assets/manifest_sha256.txt` are the only changed entries. The other 26 of its 53
entries (22 files and 4 directories) have the same content as in the scored zip (same CRC-32, size
and sha256); 5 of those 22 files (the LoRA adapter, `sandpiper_live_infer.py` and three tokenizer
files) were recompressed when the zip was rebuilt, so only their compressed bytes differ
([models/scored_zip_replaced_entries.tsv](models/scored_zip_replaced_entries.tsv) lists every
entry of both zips). The scored zip's sha256 was
`4ab1450b1b720e1dc625e73af8c51b4dd3ad331d32937d8059c14628088dba07` (DrivenData holds that file).
Check yours with `sha256sum
models/submission.zip`; it must print the `e2a8d2f2…` sha256 above. To re-pack the tree into
`build/submission_check.zip` and compare it with the zip entry by entry: `make submission` (prints
`entry check vs models/submission.zip: identical=49 different=0 missing=0 extra=0` and exits 1 on
any difference; a newly packed zip has its own sha256, because zip stores modification times).
Upload it on the competition's code-execution page.

---

# Data and licences

**Competition data: not included.** The raw data stays under the competition's terms; we do not
redistribute it, and you download it yourself (Setup step 4). Specifically:

- no file holds the transcripts, objective texts or labels. Two intermediate files of the
  pipeline consist of transcript text (`traceace_full.pkl`, `turn_text.pkl.gz`); they are not
  shipped, and steps 01a/01b regenerate them from the raw training data (`data/raw/` or the
  `--raw` folder);
- the out-of-fold prediction files in `models/predictions/oof/` ship without their label column
  `y`; `src/make_workspace.py` restores it from `train_labels.csv` (by `response_id`) and checks the
  sha256 of every restored file against the original, so the pipeline sees byte-identical files;
  `mrft_oof_3fold.csv` is rebuilt from its keyed copy;
- what remains is keyed by the competition's ids (`session_id`, `response_id`, turn number), so it
  maps onto the raw data you have;
- the trained text classifiers (the six distilled classifiers and the XGBoost model's objective-text
  vectorizers) hold TF-IDF vocabularies. As in any trained text model, single words, word pairs and
  character n-grams from the training transcripts and objective texts necessarily remain in them. We
  re-saved the seven files that contain them with every vocabulary stored in sorted order (same
  term-to-column mapping, identical transforms), so the order of the training text is not preserved.
  These seven files, 19 source files with edited comments and docstrings, and the regenerated
  `assets/manifest_sha256.txt` are the only difference between `models/submission.zip` and the
  scored zip; the training scripts now save in the same sorted form, so rebuilt classifiers are
  stored the same way (which rebuilt files are byte-identical is listed under [Run
  training](#run-training)). The XGBoost model's objective-text vectorizer keeps every term
  (`min_df=1`): 1,457 terms, of which 883 occur in only one objective
  (`src/training/scripts/train_model_e3l.py`). They are pooled and sorted, with no record of which
  objective a term came from;
- per-row predictions on the released test set ship without labels: 9 files in
  `models/predictions/test/` (the shipped model's two halves and the 8 seed replicates). They hold response ids and model outputs
  (probabilities and logits) only.

`python src/tools/audit_no_competition_data.py --raw data/raw` (`make audit`, about 10 minutes on
CPU) checks the rest. Besides the train and test files it needs the post-competition
`test_labels*.csv` (and, optionally, `qualitative_metadata*.csv`) in `data/raw/` or its `test/`
subfolder; without a required input it prints `RESULT: INCOMPLETE` and exits with status 2. It
indexes every run of 12 tokens (at least 8 of them words) of every utterance of the train and test
transcripts, every utterance of 8–11 tokens (at least 7 words) that occurs at most 3 times, as a
whole, every objective text, the train and test labels, and the ids in `qualitative_metadata.csv`.
Wording that also occurs in the pinned public files in `data/external/` (MRBench, the
national-curriculum statements) is taken out of the index. It then scans every file of the
repository except `.git/`, `build/`, `.venv*/`, `models/huggingface_models/` and `data/raw/` (apart
from its README; no subfolder is entered): every non-binary file as text, whatever its extension;
the text columns of every table (`.pkl`, `.pkl.gz`, `.parquet`); 0/1 columns and arrays (`.npy`, and
arrays inside `.joblib` files), compared by value with the labels; and every member of
`models/submission.zip` that is not byte-identical to its copy in `models/submission/`. It does
not cover utterances of fewer than 8 tokens (half of all utterances, mostly short replies), quotes
of fewer than 12 tokens taken from inside a longer utterance, or the LoRA weights (`.safetensors`).
Nor does it check strings stored inside `.joblib` files, 0/1 vectors inside JSON files, 2-D
arrays, continuous statistics derived from the labels, or embeddings.
It names files and lines but never quotes the matched text. On this repository it passed, within
that scope; its report also lists all 21 fitted TF-IDF vectorizers, every vocabulary stored in
sorted order. The hardware records (`.json`) in
`models/predictions/test/` ship verbatim. They contain a team member's cluster username and home
directory, research-tree paths, cluster host names, private 10.x addresses written by vLLM and our
workstation's host name (`parisi`); they contain no credentials.

**Licences, by path.** What the MIT licence covers, and its exceptions, is stated in the scope
paragraph at the end of [LICENSE](LICENSE). Notices: [models/llm/NOTICE.md](models/llm/NOTICE.md)
(the CC BY-SA adapters), [data/external/NOTICE.md](data/external/NOTICE.md),
[data/external/mrbench/NOTICE.md](data/external/mrbench/NOTICE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

| path | content | licence |
|---|---|---|
| `src/` (except `src/training/external/dialogue-kt/`, `src/training/docs/sandpiper_codebooks.md` and the third-party text in `src/training/scripts/sandpiper/nto_codebook.py` and `src/training/docs/39_moves_annotations_v2.md`, below), `models/submission/main.py`, `main_llmkt_lora.py`, `models/submission/assets/magnificat/` | our code | MIT ([LICENSE](LICENSE)) |
| `src/training/external/dialogue-kt/` | the dialogue-kt fork (umass-ml4ed/dialogue-kt) with our modifications | MIT under its own LICENSE (© 2024 Alexander Scarlatos for the upstream code; © 2026 team appelswim for our modifications) |
| `models/llm/lmkt_mrft_sa_8b_all35k/` (= `models/submission/assets/lora_llm/`, also inside `models/submission.zip`), `models/llm/lmkt_mrft_sa_8b_a1f1/`, `models/llm/seeds/full_s*/stage2/`, `models/llm/seeds/stage2_only_s*/` | the stage-2 LoRA adapters, trained in part on MRBench | CC BY-SA 4.0, with attribution to MRBench ([models/llm/NOTICE.md](models/llm/NOTICE.md)) |
| `models/llm/lmkt_traceace_qwen3_8b_full_{1,2,3,final}/`, `models/llm/seeds/full_s*/stage1/` | the stage-1 LoRA adapters (competition data only) | MIT |
| `models/gbm/`, `models/distilled/`, `models/calibrators/`, `models/predictions/`, and their copies in `models/submission/assets/` | our trained models and predictions | MIT |
| `models/submission/assets/tokenizer/` | the Qwen3 tokenizer files | Apache-2.0 (Qwen) |
| `data/external/nc_curriculum.csv`, `models/submission/assets/nc_curriculum.csv` | National curriculum in England: mathematics programmes of study (Department for Education) | Open Government Licence v3.0 |
| `data/external/mrbench/raw/` | MRBench, unmodified | CC BY-SA 4.0 |
| `data/external/mrbench/derived/` | our Qwen3-32B correctness labels for MRBench and the stage-2 training tables | CC BY-SA 4.0 |
| `src/training/docs/sandpiper_codebooks.md`; the definitions copied from it into `src/training/docs/39_moves_annotations_v2.md` | Sandpiper codebook text (National Tutoring Observatory) | MIT (NTO); the rest of `39_moves_annotations_v2.md` is ours (MIT) |
| `src/training/scripts/sandpiper/nto_codebook.py` | NTO Tutor Move Taxonomy, Table 1 (Zhou et al., arXiv 2603.05778), verbatim | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the surrounding code is MIT |
| `data/interim/`, `data/examples/`, `README.md` | ours | MIT |
| everything else (`Makefile`, `requirements.txt`, `MANIFEST.sha256`, `models/llm/seeds/MANIFEST.sha256`, `.gitattributes`, `.gitignore`, the other README and NOTICE files, `THIRD_PARTY_NOTICES.md`, `data/processed/.gitkeep`, and `blend_weight.txt`, `model_kind.txt`, `v2_spec.txt`, `lora_sha256.txt`, `manifest_sha256.txt` in `models/submission/assets/`) | ours | MIT |

`models/submission.zip` contains files under the licences of its paths above (MIT, CC BY-SA 4.0
for `assets/lora_llm/`, Apache-2.0 for `assets/tokenizer/`, OGL v3.0 for
`assets/nc_curriculum.csv`). The organisers confirmed on the competition forum that share-alike
licences are acceptable. For the curriculum file: *Contains public sector information licensed
under the Open Government Licence v3.0.*

Several MIT rows above are artefacts derived from the competition data: the stage-1 adapters, the
distilled classifiers and the XGBoost model (with their TF-IDF vocabularies), the predictions, and
`data/interim/`. The licence covers our work in them; it grants no rights in the competition data,
which stays under the competition's terms.

Python packages are not part of the repository; you install them from PyPI and the PyTorch index
under their own licences. Besides the tools in the table below, they pull in, among others,
NVIDIA's CUDA packages (the nvidia-* wheels and cuda-bindings / cuda-python, under NVIDIA's own
licence), certifi (MPL-2.0), tqdm (MPL-2.0 AND MIT), triton (MIT) and, in the vLLM environment,
xgrammar (Apache-2.0); THIRD_PARTY_NOTICES.md says which environments we checked.

**Third-party data, models and tools**

| asset | role | licence |
|---|---|---|
| MRBench V1, V2, V3 dev/test (Maurya, Srivatsa, Petukhova, Kochmar, "Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors", NAACL 2025; MBZUAI) | LoRA stage 2. Unmodified copies from the K-12 AI Infrastructure Platform (dataset 3, version 3: <https://platform.k12-ai-infrastructure.org/datasets/3/mrbench/version/3/>), identical to github.com/kaushal0494/UnifyingAITutorEvaluation@bbef521; sha256 in `MANIFEST.sha256` ([data/external/mrbench/NOTICE.md](data/external/mrbench/NOTICE.md)) | CC BY-SA 4.0 |
| National curriculum in England (gov.uk) | the key stage implied by the similarity-weighted mean year group of each objective's 5 closest statements (TF-IDF) is one XGBoost feature (`lo_ks_level`) | Open Government Licence v3.0 |
| `Qwen/Qwen3-8B` | base model of the LLM half (we distribute only the LoRA adapters and the tokenizer files) | Apache-2.0 |
| `Qwen/Qwen3.5-122B-A10B-FP8`, `Qwen/Qwen3-32B`, `Qwen/Qwen2.5-14B-Instruct` | offline annotators (step 02), at the revisions in the Stage 02 table | Apache-2.0 |
| GPT-4o correctness annotations of MathDial, as released with umass-ml4ed/dialogue-kt | only the reference for choosing and validating the Qwen correctness annotators (300 MathDial dialogues); never training data; not shipped | MathDial: CC BY-SA 4.0; the annotations are OpenAI model output, under OpenAI's terms |
| NTO tutor-move taxonomy (Zhou et al., arXiv 2603.05778) | codebook for 02b, transcribed in `src/training/scripts/sandpiper/nto_codebook.py`; 02e reads the same definitions from the Sandpiper copy; 15 of the 32 taxonomy-v2 tutor codes are its learning-support codes | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Sandpiper annotation codebooks and prompts (National Tutoring Observatory, [github.com/National-Tutoring-Observatory/sandpiper](https://github.com/National-Tutoring-Observatory/sandpiper)) | verbatim source of the 15 NTO move definitions in the 02e prompts; shipped as `src/training/docs/sandpiper_codebooks.md`, prompts adapted in `sandpiper_live_infer.py`. The 3 student codes `confusion_expressed`, `seeking_clarification` and `self_correcting` are ours, not Sandpiper's: we defined them for a first annotation pass that also ran the Sandpiper tutor-move prompt | MIT |
| interviews we conducted with three practising tutors (pseudonymised; `src/training/docs/29_interviews_and_labels.md` keeps paraphrased themes, not the raw notes) | the interview codebook (02d) | collected by us; MIT |
| umass-ml4ed/dialogue-kt | LLMKT training code (forked in `src/training/external/dialogue-kt`) | MIT |
| vLLM, transformers, peft, XGBoost | tooling | Apache-2.0 |
| PyTorch, scikit-learn, pandas | tooling | BSD-3-Clause |
| `Qwen/Qwen3.5-397B-A17B` | reference annotator on the external corpora; evaluation only | Apache-2.0 |
| MathDial (Macina et al., 2023) | transfer evaluation, as a separate download; also inside MRBench: 122 of the 404 stage-2 training dialogues carry MRBench's own `"Data": "MathDial"` tag (see below) | CC BY-SA 4.0 |
| QATD-2k, CoMTA, TutorChat | never training data and not shipped; used only to evaluate models | CC BY-NC 4.0 / non-commercial / unstated |
| Bridge | transfer evaluation, as a separate download; never training data in that form (MRBench contains Bridge dialogues, see below) | CC BY-NC 4.0 as distributed on the K-12 platform (/datasets/7, the copy we evaluated on) and on Hugging Face; MIT on the authors' GitHub (rosewang2008/bridge@37290e8) |

<a id="bridge-and-mathdial-inside-mrbench"></a>**Bridge and MathDial inside MRBench.** MRBench
redistributes Bridge dialogues: 37 of the 404 stage-2 training dialogues carry MRBench's own
`"Data": "Bridge"` tag, and 81 of the 404 (~20 %) match Bridge verbatim (107 of all 504 MRBench
dialogues). MRBench also redistributes MathDial dialogues (Macina et al., 2023, CC BY-SA 4.0): 122
of the 404 stage-2 training dialogues carry MRBench's `"Data": "MathDial"` tag (145 of all 504).
We used MRBench under its CC BY-SA 4.0 licence.

**Provenance stamp.** The XGBoost trainer (`src/training/scripts/train_model_e3l.py`) stamps the
booster with the corpora it was fit on, and writes a `.NEVER_SHIP.txt` marker if any of them is
non-commercial. The shipped booster's stamp is `train_corpora=('tta',)`. The distilled classifiers,
the calibrator and the LoRA adapters carry no stamp.

