.PHONY: help requirements base-model verify-repo verify-intermediates check quick-check rebuild predict example submission audit manifest manifest-check archive clean-cache clean

# Run every target from the repository root. PYTHON is the Python 3.12 environment of README
# Setup step 2 (activate .venv first, or pass PYTHON=.venv/bin/python).
PYTHON ?= python

## Show the available commands
help:
	@grep -B1 -E "^[a-z-]+:" Makefile | grep -E "^##|^[a-z-]+:" | paste - - | sed -e "s/## //" -e "s/:.*//" | awk -F"\t" '{printf "  %-22s %s\n", $$2, $$1}'

## Install the CPU / inference + rebuild environment into the active venv (README Setup step 2)
requirements:
	uv pip install --index-strategy unsafe-best-match -r src/training/requirements-cpu.txt

## Download the base model Qwen/Qwen3-8B (16 GB) at the revision the adapter was trained on
base-model:
	hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/huggingface_models/Qwen/Qwen3-8B
	bash src/tools/check_base_model.sh

## (everyone) sha256 of every shipped file against MANIFEST.sha256 (and the seed replicates, if unpacked)
verify-repo:
	sha256sum -c --quiet MANIFEST.sha256 && echo "REPOSITORY INTACT"
	@if [ -d models/llm/seeds ]; then sha256sum -c --quiet models/llm/seeds/MANIFEST.sha256 && echo "SEED REPLICATES INTACT"; fi

## (training) sha256 of every intermediate file in the pipeline's working directory
verify-intermediates:
	$(PYTHON) src/run_training.py --verify

## (training) check versions and data (assembles build/workspace on first use)
check:
	$(PYTHON) src/run_training.py --check

## (training) re-run the steps from the intermediate files to the submission (~15 min on 12 CPU threads, 12 min measured; CPU; set KT_PYTHON first)
quick-check:
	@[ -n "$$KT_PYTHON" ] || echo "note: KT_PYTHON is not set, so 07a runs under $(PYTHON) and reports DIFFERS (a and b up to 7 units in the last place away) instead of MATCH; 09a then reports 45/46 and 09b lists 2 differing entries, the calibrator and the manifest (report only, harmless). MATCH also needs an AVX-512 Intel CPU (README, Run training). README Setup step 2: [ -x .venv-kt/bin/python ] && export KT_PYTHON=\$$PWD/.venv-kt/bin/python"
	$(PYTHON) src/run_training.py --steps 01b,01c,07a,04b,04c,05b,08a,09a,09b

## (training) re-run every CPU step from the raw data (~4 h on a 6-core workstation; 09a/09b report 3 to 8 retrained assets that differ in bytes but predict identically, see README)
rebuild:
	$(PYTHON) src/run_training.py --from 01 --to 09

## Predict the competition test set in data/raw -> data/processed/submission.csv (GPU)
predict:
	$(PYTHON) src/run_inference.py

# A preflight that loads no model: the same run_inference.py command with --check added.
# The prediction must land within 0.02 of our CPU run (expected_predictions_cpu.csv, 0.649216874889283).
# Delete data/processed/example_predictions.csv before make manifest or make archive.
## Predict the one-row example in data/examples/new_data and compare it with our CPU run (CPU is enough)
example:
	$(PYTHON) src/run_inference.py --cpu --features data/examples/new_data/rows.csv --transcripts data/examples/new_data/transcripts --out data/processed/example_predictions.csv
	$(PYTHON) src/tools/score_submission.py data/processed/example_predictions.csv --reference data/examples/new_data/expected_predictions_cpu.csv --reference-column probability --max-diff 0.02

## Re-pack models/submission into build/submission_check.zip and compare it with models/submission.zip
submission:
	mkdir -p build
	bash src/tools/package_submission.sh models/submission build/submission_check.zip

## Check that the repository contains no competition data (needs the raw data incl. test labels; ~10 min; exit 0 passed, 1 files to review, 2 an input missing)
audit:
	$(PYTHON) src/tools/audit_no_competition_data.py --raw data/raw

# make manifest and make archive refuse to run on a tree with unexpected top-level entries, files in
# data/processed/ other than .gitkeep, __pycache__ folders (make clean-cache), or the placeholders
# TO FILL, LINK TO BE ADDED or [PENDING] in a shipped Markdown file; make manifest-check applies the
# first two checks only (src/tools/make_manifest_and_archive.sh).
## (maintainers) Regenerate MANIFEST.sha256 (and models/llm/seeds/MANIFEST.sha256) after editing any shipped file
manifest:
	bash src/tools/make_manifest_and_archive.sh

## (maintainers) Check that the manifests list exactly the shipped files (verify-repo checks their contents)
manifest-check:
	bash src/tools/make_manifest_and_archive.sh --check

# Needs about 10 GB free next to the output (../) and about 5 GB in $TMPDIR for the test unpack.
# Writes ../trace-the-ace_prize-winner.zip and ../trace-the-ace_prize-winner_seed-replicates.zip
# (deterministic; one top-level folder) and the sha256 of each in a .sha256 file beside it, to
# publish beside the archive (never inside it).
## (maintainers) Regenerate the manifests, pack the two release zips from exactly the files they list, test-unpack them
archive:
	PYTHON="$(PYTHON)" bash src/tools/make_manifest_and_archive.sh ../trace-the-ace_prize-winner.zip

## Delete the __pycache__ folders only (leaves build/, the virtual environments and the base model alone)
clean-cache:
	find . \( -path ./build -o -path ./.git -o -path './.venv*' -o -path ./models/huggingface_models \) -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +

## Delete build/ (the pipeline's workspace, including any retrained GPU outputs) and the __pycache__ folders; the .venv* environments are left alone
clean: clean-cache
	rm -rf build
