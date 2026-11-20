# models/

Every trained file of the solution. "Step" is the pipeline step that produces the file
(`python src/run_training.py --list`; README, "The steps"); "used" says whether the submission
(`models/submission/main.py`) reads it. Sizes are in bytes (MB = 10^6 bytes); a folder's size is
the sum of its files' sizes. The licence of every file here is MIT ([LICENSE](../LICENSE)) except
where the role column names another licence; see also [llm/NOTICE.md](llm/NOTICE.md) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Note on the seven text-classifier files

`models/submission.zip` is the scored submission with its seven text-classifier files re-saved so
that their vocabularies are stored in sorted order; predictions are identical. The seven files are
`baseline.joblib`, `doc29_clf.joblib`, `nto_moves_clf.joblib`, `stu_states_clf.joblib`,
`turn_correctness_clf.joblib`, `v2s_clf.joblib` and `v2t_clf.joblib` in `submission/assets/`, and
their copies in `distilled/` and `gbm/`. Re-saving (`src/tools/sort_vocabularies.py`) keeps every
term's column index, and `transform()` gives identical sparse matrices for every vectorizer, so
`assets/manifest_sha256.txt` was regenerated and the zip rebuilt by replacing exactly those 8
members. Later, comments and docstrings were edited in 19 source files (`main.py`,
`main_llmkt_lora.py` and 17 modules in `assets/magnificat/`; no code changed), and those members
and the manifest were replaced again. The other 26 entries (22 files + 4 directories) have identical
content (same CRC-32, size and sha256). 5 of the 22 files (the LoRA adapter,
`sandpiper_live_infer.py` and the three tokenizer files) were recompressed when the zip was rebuilt,
so only their compressed bytes differ.
[`scored_zip_replaced_entries.tsv`](scored_zip_replaced_entries.tsv) compares every entry of the two
zips. sha256 of `models/submission.zip`: `e2a8d2f28eaf3274a75413a13256feb42129ca311a538a2854078cb11dc8d2f6`;
the scored zip's sha256 was `4ab1450b1b720e1dc625e73af8c51b4dd3ad331d32937d8059c14628088dba07`
(DrivenData holds that file). Every reported score was computed with the original files and still
holds, because the transforms are identical. The training scripts now save these files the same
way (`magnificat.vocab_order.dump_sorted`). The full CPU rebuild of 2026-10-17/18 started before
that change, so its steps 03a-03e still saved vocabularies in training order. Its rebuilt
`turn_correctness_clf`, `nto_moves_clf`, `stu_states_clf` and `doc29_clf` matched the original
files, and once re-saved with sorted vocabularies (`resave_sorted`) they were byte-identical to the
shipped files. Under another Python build they differ in one field, `_stop_words_id`, which holds
`id(None)` of the interpreter that wrote them (README, Run training). `baseline.joblib` (identical boosters and out-of-fold predictions) and the two
taxonomy-v2 classifiers (identical turn probabilities) differ in bytes and predict identically.

Single words, word pairs and short character n-grams from the training text necessarily remain in
the vocabularies, as in any trained text model (the distilled classifiers' from transcripts, the
XGBoost model's from objective texts); the order of the training text is not preserved.

## submission/: the submission tree (49 files)

| path | role | step | used | size |
|---|---|---|---|---:|
| `submission/main.py` | the entry point the competition platform ran (comments and docstrings edited, code unchanged) | code | yes | 39,201 |
| `submission/main_llmkt_lora.py` | an LLM-only entry point whose model loader `main.py` copies; part of the scored zip, not called | code | no | 12,194 |
| `submission/assets/manifest_sha256.txt` | sha256 of the 46 other files in `assets/`; `main.py` checks it on startup | 09b | yes | 4,138 |
| `submission/assets/lora_llm/` | the LLM half: LoRA stage 2 (`adapter_model.safetensors`, `adapter_config.json`) = `llm/lmkt_mrft_sa_8b_all35k/`; CC BY-SA 4.0 | 06b | yes | 174,656,628 |
| `submission/assets/lora_sha256.txt` | sha256 of the two adapter files, checked before loading | 09a | yes | 178 |
| `submission/assets/llmkt_calibrator.joblib` | frozen Platt map of the LLM half = `calibrators/llmkt_calibrator_mrft_sub0808.joblib` | 07a | yes | 87 |
| `submission/assets/baseline.joblib` | the XGBoost half: 5-seed XGBoost + Platt, 177 columns = `gbm/baseline_v2shrunkD.joblib` | 05a | yes | 4,622,207 |
| `submission/assets/turn_correctness_clf.joblib` | distilled answer-correctness classifier = `distilled/turn_correctness_clf.joblib` | 03a | yes | 5,943,051 |
| `submission/assets/nto_moves_clf.joblib` | distilled NTO tutor-move classifier (its 21 `ntodst_` columns are computed but dropped by the booster) = `distilled/nto_moves_clf.joblib` | 03c | yes | 17,998,420 |
| `submission/assets/stu_states_clf.joblib` | distilled student-state classifier = `distilled/stu_states_clf.joblib` | 03d | yes | 16,671,442 |
| `submission/assets/doc29_clf.joblib` | distilled interview-codebook classifier = `distilled/doc29_clf.joblib` | 03e | yes | 13,430,612 |
| `submission/assets/v2t_clf.joblib` | distilled taxonomy-v2 tutor-move classifier = `distilled/v2t_clf_v2ext_122bC.joblib` | 03f | yes | 29,889,369 |
| `submission/assets/v2s_clf.joblib` | distilled taxonomy-v2 student-move classifier = `distilled/v2s_clf_v2stu_122bC.joblib` | 03f | yes | 11,697,028 |
| `submission/assets/nc_curriculum.csv` | National curriculum statements = `data/external/nc_curriculum.csv`; OGL v3.0 | frozen (no step) | yes | 53,295 |
| `submission/assets/blend_weight.txt` | XGBoost weight of the blend, `0.4300` (frozen) | config | yes | 7 |
| `submission/assets/model_kind.txt` | `ensemble` (LLM + XGBoost) | config | yes | 9 |
| `submission/assets/v2_spec.txt` | how the taxonomy-v2 block is built (`source=shrunk`, `narrow=none`) | config | yes | 51 |
| `submission/assets/magnificat/` | the feature and inference code (25 modules): the inference copy of 22 modules of `src/training/src/magnificat/`, some with docstrings stripped (`src/tools/check_train_serve_equivalence.py` compares the two copies), plus 3 inference-only modules (`llmkt_infer`, `model_a`, `tsl_features`) | code | yes | 308,116 |
| `submission/assets/tokenizer/` | the Qwen3 tokenizer (6 files); Apache-2.0 | upstream | yes | 15,882,422 |
| `submission.zip` | the zip of `submission/` (53 entries: 49 files + 4 directories); see the note above; its parts keep their own licences (lora_llm CC BY-SA 4.0, tokenizer Apache-2.0, nc_curriculum.csv OGL v3.0, the rest MIT) | 09b | is the submission | 263,414,059 |
| `scored_zip_replaced_entries.tsv` | entry-by-entry comparison of the scored zip (sha256 `4ab1450b…`, read from our archive copy) with `submission.zip`: CRC-32, size, compressed size and sha256 of each of the 53 entries in both, and a status (`identical` 17, `same content, recompressed` 5, `comments edited` 19, `replaced` 8, `directory` 4) | release | no | 11,044 |

## llm/: LoRA adapters of Qwen3-8B (licences: [llm/NOTICE.md](llm/NOTICE.md))

| path | role | step | used | size |
|---|---|---|---|---:|
| `llm/lmkt_mrft_sa_8b_all35k/` | stage 2, the shipped adapter (= `submission/assets/lora_llm/`), with `train_log.csv`; CC BY-SA 4.0 | 06b | yes (as `lora_llm`) | 174,656,856 |
| `llm/lmkt_traceace_qwen3_8b_full_final/` | stage 1, the parent of stage 2 | 06a | no | 174,656,323 |
| `llm/lmkt_traceace_qwen3_8b_full_1/`, `_2/`, `_3/` | stage-1 fold parents (objective-disjoint folds 1-3) | 06c | no | 3 × 174,656,323 |
| `llm/lmkt_mrft_sa_8b_a1f1/` | stage-2 continuation of fold parent 1, with `train_log.csv` (the fold-2/3 continuations were overwritten); CC BY-SA 4.0 | 06c | no | 174,656,551 |
| `llm/seeds/` | the twelve seed-replicate adapters of the next three rows, with `README.md` and `MANIFEST.sha256`; shipped in the separate archive `trace-the-ace_prize-winner_seed-replicates.zip` | | no | |
| `llm/seeds/full_s<seed>/stage1/` | seed replicates 222-225 of stage 1 (after epoch 2), with `train_log.csv` | 06a, `LLMKT_SEED` | no | 174,683,922 (s222), 3 × 174,683,914 |
| `llm/seeds/full_s<seed>/stage2/` | seed replicates of stage 2 on top of their own stage 1, with `train_log.csv`; CC BY-SA 4.0 | 06b, `LLMKT_SEED` | no | 4 × 174,657,090 |
| `llm/seeds/stage2_only_s<seed>/` | seed replicates of stage 2 on top of the shipped stage 1, with `train_log.csv`; CC BY-SA 4.0 | 06b, `LLMKT_SEED` | no | 4 × 174,657,090 |
| `llm/NOTICE.md` | licences of the adapters | | | |

## distilled/: the six distilled turn classifiers (TF-IDF + logistic regression)

| path | role | step | used | size |
|---|---|---|---|---:|
| `distilled/turn_correctness_clf.joblib` | answer correctness (2 heads) | 03a | yes (copy in assets) | 5,943,051 |
| `distilled/nto_moves_clf.joblib` | NTO tutor moves (19 heads) | 03c | yes (copy in assets) | 17,998,420 |
| `distilled/stu_states_clf.joblib` | student states (12 heads) | 03d | yes (copy in assets) | 16,671,442 |
| `distilled/doc29_clf.joblib` | interview codebook (14 heads) | 03e | yes (copy in assets) | 13,430,612 |
| `distilled/v2t_clf_v2ext_122bC.joblib` | taxonomy-v2 tutor moves (32 heads) | 03f | yes (copy in assets, as `v2t_clf.joblib`) | 29,889,369 |
| `distilled/v2s_clf_v2stu_122bC.joblib` | taxonomy-v2 student moves (12 heads) | 03f | yes (copy in assets, as `v2s_clf.joblib`) | 11,697,028 |
| `distilled/v2t_features_v2ext_122bC.meta.json`, `v2s_features_v2stu_122bC.meta.json` | metadata of the two v2 classifiers (labelled sessions and turns, mean out-of-fold AUROC) | 03f | no | 274 / 277 |

## gbm/, calibrators/

| path | role | step | used | size |
|---|---|---|---|---:|
| `gbm/baseline_v2shrunkD.joblib` | the XGBoost half (= `submission/assets/baseline.joblib`) | 05a | yes (copy in assets) | 4,622,207 |
| `calibrators/llmkt_calibrator_mrft_sub0808.joblib` | the shipped Platt map of the LLM half (= `submission/assets/llmkt_calibrator.joblib`), fit on the stage-2 out-of-fold predictions | 07a | yes (copy in assets) | 87 |

## predictions/: per-row predictions (no labels)

| path | role | step | used | size |
|---|---|---|---|---:|
| `predictions/oof/v2shrunkD_oof.csv` | out-of-fold predictions of the shipped XGBoost model (`p_raw`, `p_abl`), 35,072 rows | 05a | no | 1,606,862 |
| `predictions/oof/mrft_oof_3fold_keyed.csv` | pooled out-of-fold predictions of the three stage-2 fold models (`fold`, `p`), 16,693 rows; `make_workspace.py` rebuilds the calibrator's input `mrft_oof_3fold.csv` from it | 06c / 07b | no | 280,974 |
| `predictions/test/shipped_components.csv` | the shipped model's released-test predictions per row (`p_llm`, `p_gbm`, `p_blend`), from our 2026-08-28 H100 run of the shipped `main.py` with an 11-line block that only writes the two halves to a side file (LSF job 29267305) | inference | no | 680,713 |
| `predictions/test/seeds/<run>.csv`, `.json` | released-test predictions of the eight seed replicates (`p_raw`, `z_raw`, `p_llm`) and the hardware of each scoring run | evaluation | no | 3,972,080 (16 files) |

The two out-of-fold files ship without their label column `y`; `src/make_workspace.py` restores it
from `train_labels.csv` and checks each restored file's sha256.

## not shipped

`huggingface_models/` is where you download the base model `Qwen/Qwen3-8B` (README Setup step 3,
16 GB, Apache-2.0).

## Run tags in the file names

| tag | meaning |
|---|---|
| `lmkt`, LLMKT | LLM-based knowledge tracing, the dialogue-kt model type (`--model_type lmkt`) |
| `qwen3_8b`, `8b` | the base model Qwen3-8B |
| `traceace`, `full` | trained on the competition dialogues with the full transcript |
| `final` | trained on all folds (`--train_all`) rather than one fold |
| `mrft` | MRBench fine-tune: LoRA stage 2 |
| `all35k` | stage 2 continued from the stage-1 model trained on all ~35k competition responses |
| `a1`, `a1fK` | `a1` is the stage-1 recipe (full transcript); `a1fK` is the stage-2 continuation of fold parent K |
| `sub0808` | the shipped calibrator of the LLM half |
| `122bC` | distilled from the Qwen3.5-122B-A10B-FP8 taxonomy-v2 labels (step 02e); the letter is the run label of that annotation set |
| `v2ext`, `v2stu` | taxonomy-v2 tutor map (extended) and student map |
| `v2shrunkD` | the shipped XGBoost model: the taxonomy-v2 block as κ-shrunk rates (κ = 25) plus the interview-codebook (`doc29`) block (D is a run label) |
| `doc29` | the interview codebook, named after its document `src/training/docs/29_interviews_and_labels.md` |
| `keephdr` | header-preserving truncation of the LLM prompt at inference (`LLMKT_KEEP_HEADER=1`, set by `main.py`) |
