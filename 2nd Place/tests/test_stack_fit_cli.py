"""Stage 8 contract: fit_stack_sub9.py stops on fitting errors; --compare separates a changed
method (exit 1) from changed numbers (reported, exit 0). CPU-only, synthetic miniature
inputs, no competition data."""
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ASSETS, ROOT

SCRIPT = ROOT / "scripts" / "fit_stack_sub9.py"
N_ROWS = 35072                     # the script requires the full aligned row count


def run(*args):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def make_layout(root: Path, n: int = N_ROWS) -> Path:
    """A miniature outputs/models layout with random but valid OOF predictions."""
    rng = np.random.default_rng(0)
    ids = [f"r{i:06d}" for i in range(n)]
    y = rng.integers(0, 2, n)
    p = lambda: np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.15, n), 0.02, 0.98)
    (root / "tabular_v2_bge_no_telo").mkdir(parents=True)
    pd.DataFrame({"response_id": ids, "is_correct": y, "oof_tabular_v2": p()}).to_csv(
        root / "tabular_v2_bge_no_telo/oof.csv", index=False)
    (root / "sub5_fit").mkdir()
    pd.DataFrame({"response_id": ids, "oof_blend_cal": p()}).to_csv(
        root / "sub5_fit/oof.csv", index=False)
    folds = np.arange(n) % 5
    for run_name in ("e13_qwen4", "e13_qwen8b"):
        (root / run_name).mkdir()
        q = p()
        for k in range(5):
            m = folds == k
            pd.DataFrame({"response_id": np.array(ids)[m], "p_llm": q[m]}).to_csv(
                root / run_name / f"p_fold{k}.csv", index=False)
    return root


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    return make_layout(tmp_path_factory.mktemp("oof"))


@pytest.fixture
def broken(layout, tmp_path):
    """A private copy of the layout to break."""
    import shutil
    dst = tmp_path / "oof"
    shutil.copytree(layout, dst)
    return dst


# ------------------------------------------------------------------ stage 8 in run_training.sh
def test_stage8_has_no_error_suppression():
    text = (ROOT / "scripts/run_training.sh").read_text()
    block = re.search(r"^stage8\(\) \{.*?^\}", text, re.S | re.M).group(0)
    for pat in (r"\|\|\s*echo", r"\|\|\s*true", r"\|\|\s*:", r"set\s+\+e"):
        assert not re.search(pat, block), f"stage 8 suppresses errors: {pat}"
    assert "--compare" in block and "--check" not in block


# ------------------------------------------------------------------ fit errors
def test_valid_miniature_fits(layout, tmp_path):
    r = run("--models", layout, "--out", tmp_path / "s.json")
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((tmp_path / "s.json").read_text())["features"][0] == "z_blend"


def test_missing_oof_file_fails_and_names_it(broken, tmp_path):
    victim = broken / "e13_qwen8b" / "p_fold3.csv"
    victim.unlink()
    r = run("--models", broken, "--out", tmp_path / "s.json")
    assert r.returncode != 0
    assert "missing OOF file" in r.stderr and str(victim) in r.stderr
    assert not (tmp_path / "s.json").exists()


def test_empty_fold_fails(broken, tmp_path):
    f = broken / "e13_qwen4" / "p_fold2.csv"
    f.write_text("response_id,p_llm\n")
    r = run("--models", broken, "--out", tmp_path / "s.json")
    assert r.returncode != 0 and "no rows" in r.stderr and "p_fold2.csv" in r.stderr


def test_mismatched_ids_fail(broken, tmp_path):
    f = broken / "sub5_fit" / "oof.csv"
    df = pd.read_csv(f)
    df.loc[0, "response_id"] = "not-a-real-row"
    df.to_csv(f, index=False)
    r = run("--models", broken, "--out", tmp_path / "s.json")
    assert r.returncode != 0 and "do not match" in r.stderr


def test_invalid_labels_fail(broken, tmp_path):
    f = broken / "tabular_v2_bge_no_telo" / "oof.csv"
    df = pd.read_csv(f)
    df.loc[5, "is_correct"] = 2
    df.to_csv(f, index=False)
    r = run("--models", broken, "--out", tmp_path / "s.json")
    assert r.returncode != 0 and "is_correct must be 0/1" in r.stderr


def test_nan_probabilities_fail(broken, tmp_path):
    f = broken / "e13_qwen8b" / "p_fold0.csv"
    df = pd.read_csv(f)
    df.loc[3, "p_llm"] = np.nan
    df.to_csv(f, index=False)
    r = run("--models", broken, "--out", tmp_path / "s.json")
    assert r.returncode != 0 and "NaN" in r.stderr and "p_fold0.csv" in r.stderr


def test_write_failure_fails(layout, tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    r = run("--models", layout, "--out", blocker / "llm_stack.json")
    assert r.returncode != 0 and "cannot write" in r.stderr


# ------------------------------------------------------------------ --compare
@pytest.fixture
def shipped():
    return json.loads((ASSETS / "llm_stack.json").read_text())


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, indent=2))
    return p


def test_compare_identical_copies_match(tmp_path, shipped):
    r = run("--compare", write(tmp_path, "a.json", shipped), write(tmp_path, "b.json", shipped))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MATCH" in r.stdout and "DIFFERS" not in r.stdout


def test_compare_shipped_against_itself_in_place():                  # the real file
    r = run("--compare", ASSETS / "llm_stack.json", ASSETS / "llm_stack.json")
    assert r.returncode == 0 and "MATCH" in r.stdout


def test_compare_changed_coefficient_differs(tmp_path, shipped):
    new = copy.deepcopy(shipped)
    new["coef"][1] += 0.01
    r = run("--compare", write(tmp_path, "new.json", new), write(tmp_path, "ref.json", shipped))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DIFFERS" in r.stdout
    assert re.search(r"parameter\s+NEW\s+REF\s+\|diff\|", r.stdout)
    assert re.search(r"z_tabv2\s+\S+\s+\S+\s+1\.00e-02", r.stdout)
    assert "max |diff| 1.00e-02" in r.stdout


@pytest.mark.parametrize("change", ["features", "manual_blend_w", "clip", "bundles", "ratios"])
def test_compare_structure_change_fails(tmp_path, shipped, change):
    new = copy.deepcopy(shipped)
    if change == "features":
        new["features"] = list(reversed(new["features"]))
    elif change == "manual_blend_w":
        new["manual_blend"][0]["w"] = 0.2
    elif change == "clip":
        new["clip"] = [0.02, 0.98]
    elif change == "bundles":
        new["bundles"] = new["bundles"][:2]
    else:
        new["provenance"] = {"ratios": {"z_ft_4b": 1.0, "z_ft_8b": 1.0}}
    r = run("--compare", write(tmp_path, "new.json", new), write(tmp_path, "ref.json", shipped))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STRUCTURE MISMATCH" in r.stdout


@pytest.mark.parametrize("content", ["{not json", "[]", '{"features": ["a"], "coef": [], "intercept": 0}'])
def test_compare_invalid_json_fails(tmp_path, shipped, content):
    bad = tmp_path / "bad.json"
    bad.write_text(content)
    r = run("--compare", bad, write(tmp_path, "ref.json", shipped))
    assert r.returncode == 1


def test_compare_does_not_fit(tmp_path, shipped):
    """--compare must work without any OOF predictions present."""
    r = run("--models", tmp_path / "nowhere", "--compare",
            write(tmp_path, "a.json", shipped), write(tmp_path, "b.json", shipped))
    assert r.returncode == 0 and "MATCH" in r.stdout
