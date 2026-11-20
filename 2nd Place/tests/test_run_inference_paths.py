"""scripts/run_inference.sh path handling, on a throwaway tree with a stub main.py: a
relative OUT_DIR (or DATA_DIR) is resolved against the caller's directory, and nothing is
left under inference/. CPU-only, no weights, no data."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

STUB_MAIN = """\
import os, pathlib
pathlib.Path("data/test_features.csv").read_text()          # reads ./data like the real one
pathlib.Path("submission.csv").write_text("response_id,probability\\nr1,0.5\\n")
print("stub main.py ran in", os.getcwd())
"""
STUB_VERIFY = """\
import pathlib
pathlib.Path(__file__).with_name("verify_ran").write_text("yes")   # marker for the tests
print("all 119 files of the scored archive verified (size + CRC32)")
"""


@pytest.fixture
def tree(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "run_inference.sh", repo / "scripts" / "run_inference.sh")
    (repo / "scripts" / "download_weights.py").write_text(STUB_VERIFY)
    (repo / "inference").mkdir()
    (repo / "inference" / "main.py").write_text(STUB_MAIN)
    work = tmp_path / "work"
    data = work / "mydata"
    (data / "test_transcripts").mkdir(parents=True)
    (data / "test_features.csv").write_text("response_id,session_id\nr1,s1\n")
    (data / "submission_format.csv").write_text("response_id,probability\nr1,0.5\n")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (shim / "python").chmod(0o755)
    stage = work / "stage"                       # a retrained tree (outputs/staging layout)
    (stage / "assets").mkdir(parents=True)
    (stage / "main.py").write_text(STUB_MAIN)
    (stage / "assets" / "llm_stack.json").write_text("{}")
    return {"repo": repo, "work": work, "stage": stage, "env": {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}",
                                                 "PYTHONDONTWRITEBYTECODE": "1"}}


def run(tree, **env):
    e = {**tree["env"], **env}
    if "OUT_DIR" not in env:
        e.pop("OUT_DIR", None)
    return subprocess.run(["bash", str(tree["repo"] / "scripts" / "run_inference.sh")],
                          cwd=tree["work"], env=e, capture_output=True, text=True)


def inference_listing(tree):
    inf = tree["repo"] / "inference"
    return sorted(str(p.relative_to(inf)) for p in inf.rglob("*"))


@pytest.mark.parametrize("out_dir", ["out_rel", "nested/out", "./out_dot"])
def test_relative_out_dir_resolves_against_the_caller(tree, out_dir):
    r = run(tree, DATA_DIR="mydata", OUT_DIR=out_dir)          # both relative
    assert r.returncode == 0, r.stdout + r.stderr
    out = (tree["work"] / out_dir).resolve()
    assert (out / "submission.csv").read_text().startswith("response_id,probability")
    assert "stub main.py ran in" in (out / "log.txt").read_text()
    assert f"-> {out}/submission.csv" in r.stdout              # the reported path is absolute
    assert inference_listing(tree) == ["main.py"]              # nothing written under inference/


def test_absolute_out_dir(tree, tmp_path):
    out = tmp_path / "abs_out"
    r = run(tree, DATA_DIR=str(tree["work"] / "mydata"), OUT_DIR=str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "submission.csv").is_file() and (out / "log.txt").is_file()
    assert inference_listing(tree) == ["main.py"]


def test_default_out_dir_is_repo_submission(tree):
    r = run(tree, DATA_DIR="mydata")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tree["repo"] / "submission" / "submission.csv").is_file()
    assert inference_listing(tree) == ["main.py"]


def test_the_real_script_resolves_out_dir_before_cd():
    text = (ROOT / "scripts" / "run_inference.sh").read_text()
    resolve = text.index('OUT_DIR="$(cd "${OUT_DIR:-$ROOT/submission}" && pwd)"')
    assert resolve < text.index('cd "$INFERENCE_DIR"')


def verify_ran(tree):
    return (tree["repo"] / "scripts" / "verify_ran").exists()


def test_default_inference_dir_is_verified(tree):
    r = run(tree, DATA_DIR="mydata", OUT_DIR="out")
    assert r.returncode == 0, r.stdout + r.stderr
    assert verify_ran(tree) and "verified" in r.stdout


def test_explicit_shipped_inference_dir_is_still_verified(tree):
    r = run(tree, DATA_DIR="mydata", OUT_DIR="out", INFERENCE_DIR="../repo/inference")
    assert r.returncode == 0, r.stdout + r.stderr
    assert verify_ran(tree) and "skipping" not in r.stdout


def test_relative_retrained_inference_dir(tree):
    r = run(tree, DATA_DIR="mydata", OUT_DIR="out", INFERENCE_DIR="stage")    # relative
    assert r.returncode == 0, r.stdout + r.stderr
    assert not verify_ran(tree)                                               # check skipped
    assert "retrained tree, skipping the scored-archive weights check" in r.stdout
    assert f"INFERENCE_DIR={tree['stage']}" in r.stdout                        # resolved, absolute
    out = tree["work"] / "out"
    assert (out / "submission.csv").is_file()
    assert f"stub main.py ran in {tree['stage']}" in (out / "log.txt").read_text()
    assert not (tree["stage"] / "data").exists() and not (tree["stage"] / "submission.csv").exists()
    assert inference_listing(tree) == ["main.py"]              # repo inference/ untouched


@pytest.mark.parametrize("missing", ["main.py", "assets/llm_stack.json"])
def test_retrained_tree_needs_main_and_stack(tree, missing):
    (tree["stage"] / missing).unlink()
    r = run(tree, DATA_DIR="mydata", OUT_DIR="out", INFERENCE_DIR="stage")
    assert r.returncode == 2 and f"{missing} not found" in r.stdout
    assert not verify_ran(tree) and inference_listing(tree) == ["main.py"]


def test_the_real_script_resolves_inference_dir_before_cd():
    text = (ROOT / "scripts" / "run_inference.sh").read_text()
    resolve = text.index('INFERENCE_DIR="$(cd "${INFERENCE_DIR:-$ROOT/inference}" && pwd)"')
    assert resolve < text.index('cd "$INFERENCE_DIR"')
    assert 'cd "$ROOT/inference"\n' not in text
