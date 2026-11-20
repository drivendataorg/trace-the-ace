"""Shared fixtures. All tests are CPU-only and need no weights or competition data;
tests that do need them skip themselves."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# Importing inference/main.py must not leave __pycache__ inside the scored tree.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
INFERENCE = ROOT / "inference"
ASSETS = INFERENCE / "assets"
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def manifest():
    return json.loads((ROOT / "models" / "weights_manifest.json").read_text())


@pytest.fixture(scope="session")
def stack():
    return json.loads((ASSETS / "llm_stack.json").read_text())


@pytest.fixture(scope="session")
def inference_main():
    """inference/main.py as a module (its import has no side effects beyond env defaults)."""
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location("ace_inference_main", INFERENCE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
