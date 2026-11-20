"""Training stage 4 must hand build_student_bge_coverage a loaded encoder, never a str
(a str has .encode, which the shared helper takes for a loaded model).
CPU-only: torch and sentence-transformers are replaced by fakes.
"""
import sys
import types

import pytest

from ace.models import build_tabular_v2_features as b


class FakeST:
    def __init__(self, name, device=None):
        self.name, self.device = name, device

    def encode(self, texts, **kw):
        raise AssertionError("not called here")


@pytest.fixture
def fakes(monkeypatch):
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = FakeST
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_load_bge_encoder_returns_a_model_not_a_str(fakes):
    m = b.load_bge_encoder("models/hf/bge-base-en-v1.5")
    assert isinstance(m, FakeST) and not isinstance(m, str)
    assert m.name == "models/hf/bge-base-en-v1.5" and m.device == "cpu"
    assert b.load_bge_encoder("x", "cuda").device == "cuda"


def test_build_script_passes_the_loaded_encoder():
    src = open(b.__file__, encoding="utf-8").read()
    assert "model_name_or_path=load_bge_encoder(args.bge_model, args.device)" in src
    assert "model_name_or_path=args.bge_model" not in src
