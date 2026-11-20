#!/bin/bash
set -ex
P=/work/magnificat/venv/bin/pip
$P install -U pip wheel setuptools
# torch with cu128 (Blackwell sm_100 support for the B200 job)
$P install torch --index-url https://download.pytorch.org/whl/cu128
# transformers / peft / accelerate pins
$P install "transformers==5.12.1" "peft==0.19.1" "accelerate==1.14.0"
# rest of the dialogue-kt stack
$P install "datasets==5.0.0" "pandas==2.3.3" "numpy==2.2.6" "scikit-learn==1.7.2" \
  "sentence-transformers==5.6.0" "krippendorff==0.8.2" "openai==2.44.0" "wandb==0.28.0" \
  sentencepiece tqdm matplotlib scipy
# KT baselines imported at module level; --no-deps so they cannot downgrade torch/pandas
$P install --no-deps "pyBKT==1.4.1" "pykt-toolkit==0.0.38"
echo SETUP_DONE
