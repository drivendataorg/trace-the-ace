"""Re-derive shipped assets from the original out-of-fold outputs, when available.

The original predictions contain competition data and are not distributed. Point
ACE_OOF_DIR at a folder with the outputs/models layout (sub5_fit/, tabular_v2_bge_no_telo/,
e13_qwen4/, e13_qwen8b/) to run these; otherwise they skip.

The W050 stack test needs only the OOF predictions. The tabular_v2 byte-identity test also
needs the 15 trained boosters (tabular_v2_bge_no_telo/models/seed*_fold*.txt): it skips when
none is present and fails when only some are.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ASSETS, ROOT

OOF = Path(os.environ["ACE_OOF_DIR"]) if os.environ.get("ACE_OOF_DIR") else None
needs_oof = pytest.mark.skipif(OOF is None, reason="ACE_OOF_DIR not set")
TABV2 = "tabular_v2_bge_no_telo"
BOOSTERS = [f"seed{s}_fold{f}.txt" for s in (42, 1337, 2024) for f in range(5)]
ABSENT_REASON = ("tabular_v2 boosters not provided (the private reviewer bundle has OOF "
                 "predictions and configs, not the 15 training-output boosters); "
                 "byte-identity check skipped")


def booster_status(model_dir: Path) -> tuple[str, list[str]]:
    """("absent" | "partial" | "complete", missing booster names) under <model_dir>/models/."""
    missing = [b for b in BOOSTERS if not (Path(model_dir) / "models" / b).is_file()]
    if len(missing) == len(BOOSTERS):
        return "absent", missing
    return ("partial", missing) if missing else ("complete", [])


def run(*args):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True)


@needs_oof
def test_stage_tabular_v2_is_byte_identical(tmp_path):
    state, missing = booster_status(OOF / TABV2)
    if state == "absent":
        pytest.skip(ABSENT_REASON)
    if state == "partial":
        pytest.fail(f"{OOF / TABV2 / 'models'}: {len(missing)} of 15 tabular_v2 boosters "
                    f"missing: {', '.join(missing)}")
    r = run("scripts/stage_tabular_v2.py", "--model-dir", OOF / "tabular_v2_bge_no_telo",
            "--out", tmp_path / "tabular_v2", "--check", ASSETS / "tabular_v2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "byte-identical" in r.stdout


@needs_oof
def test_fit_stack_reproduces_coefficients(tmp_path):
    pytest.importorskip("sklearn")
    r = run("scripts/fit_stack_sub9.py", "--models", OOF, "--out", tmp_path / "llm_stack.json",
            "--check", ASSETS / "llm_stack.json", "--atol", "1e-9")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("n,state,n_missing", [(0, "absent", 15), (7, "partial", 8),
                                               (15, "complete", 0)])
def test_booster_status(tmp_path, n, state, n_missing):
    (tmp_path / "models").mkdir()
    for b in BOOSTERS[:n]:
        (tmp_path / "models" / b).write_text("tree\n")
    got, missing = booster_status(tmp_path)
    assert got == state and len(missing) == n_missing
    assert missing == BOOSTERS[n:]                       # the names, in order
