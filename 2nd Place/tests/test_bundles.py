"""The three reader bundles resolve the way vLLM 0.21 needs them (no weights required)."""
import json

import pytest

from conftest import ASSETS
from ace.llm import bundle as B


@pytest.mark.parametrize("name,arch,lora,lm_only", [
    ("llm_4b", "Qwen3ForCausalLM", True, False),
    ("llm_8b", "Qwen3ForCausalLM", True, False),
    ("llm_q35", "Qwen3_5ForConditionalGeneration", False, True),
])
def test_bundle_resolves(name, arch, lora, lm_only):
    d = ASSETS / name
    meta = B.validate_bundle(d)                      # structural pre-flight, incl. architecture
    assert meta.uses_lora is lora
    assert meta.language_model_only is lm_only
    base = B.resolve_base_path(d, meta)
    assert base == d / "base"                        # embedded, relative to the bundle
    assert B.read_base_architectures(base) == [arch]
    kw = B.engine_kwargs(meta, model_path=base, max_input_tokens=8192, enforce_eager=False)
    assert kw["max_model_len"] == 8192 + B.GEN_MARGIN_TOKENS
    assert kw["enable_lora"] is lora
    assert ("language_model_only" in kw) is lm_only
    if lora:
        assert [p.name for p in B.adapter_dirs(d)] == ["full"]
        assert kw["max_lora_rank"] == 16 and "dtype" not in kw   # Qwen3 engine kwargs: no dtype
    else:
        assert B.adapter_dirs(d) == [] and kw["dtype"] == "bfloat16"


def test_qwen3_bundles_are_legacy():
    for name in ("llm_4b", "llm_8b"):
        assert not (ASSETS / name / "bundle.json").exists()
        assert B.load_bundle_meta(ASSETS / name).source == "default"


def test_q35_bundle_json():
    meta = B.load_bundle_meta(ASSETS / "llm_q35")
    assert meta.is_qwen35 and meta.adapter_mode == "dense" and meta.base_mode == "embedded"
    assert meta.expected_architecture == B.QWEN35_VLLM_ARCHITECTURE
    assert meta.folds == ["full"] and not meta.incomplete_cv


def test_bundle_metadata_has_no_absolute_paths():
    for p in list(ASSETS.glob("llm_*/bundle.json")) + list(ASSETS.glob("llm_*/base/dense_export_manifest.json")):
        text = p.read_text()
        for s in json.loads(text).values():
            if isinstance(s, str):
                assert not s.startswith(("/", "~")) and ":\\" not in s, (p, s)


def test_training_arch_is_rejected(tmp_path):
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "config.json").write_text(json.dumps(
        {"architectures": [B.QWEN35_TRAINING_ARCHITECTURE]}))
    (tmp_path / "bundle.json").write_text((ASSETS / "llm_q35" / "bundle.json").read_text())
    with pytest.raises(B.BundleValidationError):
        B.validate_bundle(tmp_path)
