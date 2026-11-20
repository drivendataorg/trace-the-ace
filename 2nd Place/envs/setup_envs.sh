#!/usr/bin/env bash
# Create the virtual environments with uv (https://docs.astral.sh/uv/).
#   bash envs/setup_envs.sh inference            # enough to reproduce the submission
#   bash envs/setup_envs.sh tabular llm qwen35   # training environments
# Each env lands in .venv-<name>/ at the repo root; activate with: source .venv-<name>/bin/activate
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[[ $# -gt 0 ]] || set -- inference
for name in "$@"; do
  req="envs/requirements-${name}.txt"
  [[ -f "$req" ]] || { echo "unknown env '$name' (expected one of: inference tabular llm qwen35)"; exit 2; }
  echo "== creating .venv-${name} from ${req}"
  # uv's managed CPython ships the headers (Python.h) Triton needs to JIT-compile its
  # CUDA helper at the first vLLM start, so no system python3.12-dev is required.
  # UV_VENV_CLEAR=1: a re-run replaces an existing .venv-<name> instead of failing.
  UV_VENV_CLEAR=1 uv venv --python 3.12 --python-preference only-managed ".venv-${name}"
  VIRTUAL_ENV=".venv-${name}" uv pip install --index-strategy unsafe-best-match -r "$req"
  VIRTUAL_ENV=".venv-${name}" uv pip install --no-deps -e .   # the `ace` package (src/)
done
