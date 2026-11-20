"""The scored inference code and the training package are the same files."""
from pathlib import Path

import pytest

from conftest import ASSETS, INFERENCE, ROOT

MIRRORED = sorted(p.relative_to(ASSETS / "ace") for p in (ASSETS / "ace").rglob("*.py"))


def test_main_py_is_identical():
    assert (INFERENCE / "main.py").read_bytes() == \
        (ROOT / "src/ace/submission/main.py").read_bytes()


def test_mirror_is_complete():
    assert len(MIRRORED) == 13


@pytest.mark.parametrize("rel", MIRRORED, ids=str)
def test_mirrored_module_is_identical(rel: Path):
    assert (ASSETS / "ace" / rel).read_bytes() == (ROOT / "src/ace" / rel).read_bytes()
