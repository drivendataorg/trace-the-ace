# Trace the Ace — LLM-KT Qwen3.5-9B (richer judge)

Predicts whether a student answers their post-tutoring quiz question correctly, given the tutoring-session transcript and the learning objective (competition metric: log loss). A LoRA-tuned Qwen3.5-9B knowledge-tracing model scores each (session, objective) pair, aided by a fine-tuned 2B judge that rates transcript evidence; this build scored 0.5964 on the public leaderboard.

## Setup

1. Download the kt adapter from [kt_adapter](https://kookree-my.sharepoint.com/:u:/g/personal/nicholas_kookree_ai/IQDbeVckH6mbRZ3CI7Qhwf5JAdbVtYpPi4DKhw4jb_Oslf8?e=BjJz29) and extract into `kt_adapter_s0/`
2. Download the model  from [Model](https://kookree-my.sharepoint.com/:u:/g/personal/nicholas_kookree_ai/IQDUgpbg2zMCQYhi04spE25HAQyRByVApTWpRJZbbkKqm20?e=oZxhm3) and extract into `models/`

   ```
   models/
   ├── assets/          # bundled feature mapper, clusters, standards, and constants
   ├── kt9b_base/        # Qwen3.5-9B base (4 safetensors shards)
   ├── judge2b/          # fine-tuned 2B judge
   ├── minilm/           # sentence-embedding model
   └── kt_adapters.zip   # LoRA adapter (s0/), loaded from the zip at runtime
   ```

   (`kt_adapter_s0/` is a reference copy of the raw LoRA adapter; the code only reads `models/kt_adapters.zip`.)

3. Install Python **3.14.4**, then create and activate a virtual environment and install dependencies:

   ```bash
   python3.14 -m venv venv
   source venv/bin/activate
   python -m pip install -r requirements.txt
   ```

   The current local environment uses Python **3.14.4** and PyTorch **2.14.1** (pinned in `requirements.txt`). These versions have not yet been validated with a complete model inference run.

## Inference

Place the input data in a `data/` folder at the repository root:

```
data/
├── test_features.csv
├── submission_format.csv
└── test_transcripts/<session_id>.csv   # one transcript per session
```

Then, from the repository root:

```
python3 src/run_inference.py
```

Predictions are written to `submission.csv`. Requires a CUDA GPU with ~24 GB+ VRAM (9B in BF16 + 2B judge); all models load from the local `models/` dir — no network access needed at runtime.

## Training from raw data

All Python scripts and reference resources are in `src/`. Start with these inputs:

```text
data/
├── train_features_TMQTWsB.csv
├── train_labels_44ujmj2.csv
└── transcripts/<session_id>.csv
```

Features need `response_id`, `session_id`, `learning_objective_id`, and `learning_objective`. Labels need `response_id` and `is_correct` (0 or 1). Transcripts need `role` and `content`; `timestamp` is optional.

Install the dependencies above and place the pretrained models in `models/kt9b_base/`, `models/judge2b_base/` (the original Qwen3.5-2B base), and `models/minilm/`. The existing fine-tuned `models/judge2b/` is used for inference. CCSS reference descriptions are included in `src/resources/ccss_hier.json`.

### 1. Prepare data and human annotations

```bash
python3 src/run_training.py prepare
```

This generates TIM features, folds, tutor–student pairs, embedding clusters, objective-to-skill mappings, and two annotation sheets in `data/`:

- **`turn_labels.csv`**: review the exchange and preceding context, then fill `verdict` with `true` (correct answer), `false` (incorrect or absent answer to an assessing question), or `na` (not an assessment or too unclear to judge). These human annotations supervise the 2B judge. Blank rows are skipped; annotate multiple sessions so training and validation are separate.
- **`cluster_labels.csv`**: review each cluster's examples and fill `kc_codes` with matching CCSS codes, comma-separated or as a JSON list. Use `[]` when no skill applies. The generated `ccss_standards.csv` lists valid codes and descriptions. Every cluster needs a decision.

The annotation sheets are preserved when preparation is rerun. Intermediate files are generated automatically; no existing folds, clusters, judge probabilities, or trained feature assets are required. Quiz outcomes supervise the 9B model and feature mapper, and are never substituted for human exchange verdicts.

### 2. Train

```bash
python3 src/run_training.py all --epochs 2 --micro 8
```

This validates the completed human annotations, trains the 2B judge, scores all assessable pairs, fits the 27 TIM + 7 mastery feature mapper, and trains the 9B LoRA adapter. GPU training requires CUDA; lower `--micro` to reduce the 9B batch size. Results go to `runs/judge/`, `runs/features/`, and `runs/kt/`.

To train one component, replace `all` with `kt`, `judge`, or `mapper`. `kt` requires only raw data and its base model; it does not require human annotations. `mapper` uses the latest completed judge checkpoint. Use `--data-dir`, `--transcripts`, or `--output-dir` for other locations. If raw data changes after annotation, use a fresh data directory to keep annotations aligned with the correct exchanges and clusters.

New checkpoints and assets stay under `runs/`. To use them for inference, validate/select a KT checkpoint and package its adapter as `s0/` inside `models/kt_adapters.zip`; merge the selected judge adapter with its base and save the merged model and tokenizer in `models/judge2b/`; copy the feature assets into `models/assets/`. Exported calibration/blend values are starting defaults and must be refitted for the new models. Full GPU training has not been validated in the local environment.
