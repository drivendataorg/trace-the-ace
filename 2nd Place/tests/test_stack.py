"""The W050 stack: main.py's blend helpers give exactly the README formula."""
import numpy as np
import pytest


def readme_formula(z):
    z8 = 0.5 * z["z_ft_8b"] + 0.5 * z["z_ft_q35"]
    s = (-0.1077 + 0.2669 * z["z_blend"] + 0.1945 * z["z_tabv2"]
         + 0.1511 * z["z_ft_4b"] + 0.4209 * z8)
    return np.clip(1 / (1 + np.exp(-s)), 0.01, 0.99)


def shipped(main, stack, z):
    """The stack lines of main.main(), using its own helpers."""
    vals = {k: v.copy() for k, v in z.items()}
    main.apply_manual_blends(vals, main.manual_blends(stack))
    s = float(stack["intercept"]) + sum(
        c * vals[f] for c, f in zip(stack["coef"], stack["features"]))
    return np.clip(1 / (1 + np.exp(-s)), *stack["clip"])


@pytest.fixture
def logits():
    rng = np.random.default_rng(0)
    return {k: rng.normal(0, 1.5, 5000)
            for k in ("z_blend", "z_tabv2", "z_ft_4b", "z_ft_8b", "z_ft_q35")}


def test_stack_shape(stack):
    assert stack["features"] == ["z_blend", "z_tabv2", "z_ft_4b", "z_ft_8b"]
    assert stack["manual_blend"] == [{"target": "z_ft_8b", "source": "z_ft_q35", "w": 0.5}]
    assert stack["clip"] == [0.01, 0.99]
    assert {b["feature"]: b["dir"] for b in stack["bundles"]} == {
        "z_ft_4b": "llm_4b", "z_ft_8b": "llm_8b", "z_ft_q35": "llm_q35"}


def test_manual_blend_parsing(inference_main, stack):
    assert inference_main.manual_blends(stack) == [
        {"target": "z_ft_8b", "source": "z_ft_q35", "w": 0.5}]
    assert inference_main.manual_blends({"features": ["a"]}) == []
    with pytest.raises(RuntimeError):
        inference_main.manual_blends({"features": ["a"], "manual_blend": [
            {"target": "a", "source": "b", "w": 1.5}]})
    with pytest.raises(RuntimeError):
        inference_main.manual_blends({"features": ["a"], "manual_blend": [
            {"target": "x", "source": "b", "w": 0.5}]})


def test_apply_manual_blend_is_the_convex_mix(inference_main, stack, logits):
    vals = {k: v.copy() for k, v in logits.items()}
    inference_main.apply_manual_blends(vals, inference_main.manual_blends(stack))
    np.testing.assert_allclose(vals["z_ft_8b"], 0.5 * logits["z_ft_8b"] + 0.5 * logits["z_ft_q35"])
    np.testing.assert_array_equal(vals["z_ft_q35"], logits["z_ft_q35"])
    with pytest.raises(RuntimeError):   # a declared source that was not scored is an error
        inference_main.apply_manual_blends({"z_ft_8b": logits["z_ft_8b"]},
                                           inference_main.manual_blends(stack))


def test_shipped_stack_equals_readme_formula(inference_main, stack, logits):
    p_ship, p_readme = shipped(inference_main, stack, logits), readme_formula(logits)
    # README rounds the coefficients to 4 decimals
    assert np.abs(p_ship - p_readme).max() < 2e-3
    assert p_ship.min() >= 0.01 and p_ship.max() <= 0.99


def test_coefficients_carry_the_logit_scale_ratios(stack):
    raw_4b, raw_8b = 0.17643278322662082, 0.42860461879301565   # raw OOF fit (stack note)
    c = dict(zip(stack["features"], stack["coef"]))
    assert c["z_ft_4b"] == pytest.approx(raw_4b * 0.8567, abs=1e-12)
    assert c["z_ft_8b"] == pytest.approx(raw_8b * 0.9821, abs=1e-12)
