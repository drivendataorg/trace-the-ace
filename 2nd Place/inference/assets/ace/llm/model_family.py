#!/usr/bin/env python
"""Model-family detection and loading shared by train_lora.py and score_hf.py (E23).

Why this exists: Qwen3.5 needs a different load path from every reader ACE has
trained so far — the text-only ``Qwen3_5ForCausalLM`` rather than the conditional-
generation/vision wrapper `AutoModelForCausalLM` may select, and SDPA rather than
the try-flash-attention-2-then-fall-back dance. Training and scoring must agree on
that choice exactly, or the OOF predictions describe a different model than the one
trained. One helper, used by both, is how they stay in agreement.

Everything here is deliberately import-light: `transformers`/`torch` are imported
inside the functions that need them, so family detection and the capability checks
stay unit-testable in an environment with neither installed.

The generic (non-Qwen3.5) path reproduces train_lora.py's historical behavior
byte-for-byte, including the flash-attention-2 attempt and its SDPA fallback. Every
Qwen3/Phi-4/OLMo3 adapter trained to date came off that path and must stay
reproducible.
"""
from __future__ import annotations

import inspect

QWEN35 = "qwen3_5"
GENERIC = "generic"


def _norm(text) -> str:
    """Normalize a model_type/architecture token for family matching.

    Hub configs spell the family as ``qwen3_5``, but ``qwen3.5``/``qwen3-5`` show up
    in hand-written configs and local checkpoints, so fold the separators together.
    """
    return str(text).lower().replace(".", "_").replace("-", "_")


def _mentions_qwen35(text) -> bool:
    """True for qwen3_5, qwen3_5_text, qwen3_5_moe, Qwen3_5ForCausalLM, ...

    Substring rather than equality so text-only and MoE variants match, and safe
    against the legacy family because ``qwen3`` never contains ``qwen3_5``.
    """
    return QWEN35 in _norm(text)


def resolve_model_family(config) -> str:
    """Return QWEN35 or GENERIC for a loaded AutoConfig.

    Checks the outer ``model_type``, any nested ``text_config.model_type`` (how
    multimodal configs carry the language tower), and the ``architectures`` list —
    a saved text-only checkpoint may report only one of the three. Deliberately not
    a substring test on the user-supplied path: a local directory named
    ``.../qwen3_5-experiment`` holding Qwen3 weights must not be misrouted.
    """
    for value in (getattr(config, "model_type", None),
                  getattr(getattr(config, "text_config", None), "model_type", None)):
        if value and _mentions_qwen35(value):
            return QWEN35
    for arch in (getattr(config, "architectures", None) or []):
        if _mentions_qwen35(arch):
            return QWEN35
    return GENERIC


def detect_family(model_path: str) -> tuple[str, object]:
    """(family, config) for a model path/id, without loading any weights."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    return resolve_model_family(config), config


def can_use_last_token_logits(ans_pos, seq_len) -> bool:
    """True when every row's answer position IS the final tensor position.

    Only then does ``logits_to_keep=1`` (which returns logits for the last position
    of the padded tensor) address the same logits as ``gather(ans_pos)``. With the
    right-padded collator this holds for batch=1, and for any batch whose rows share
    a length — but a variable-length batch leaves short rows pointing into padding,
    where the two are NOT equivalent and the full-logits path must be used.

    Takes a plain length rather than the tensor so it stays torch-free and testable.
    """
    last = int(seq_len) - 1
    return all(int(p) == last for p in ans_pos)


def supported_init_fields(cls) -> set[str]:
    """Names ``cls`` accepts at construction — dataclass fields and/or __init__ args."""
    names: set[str] = set()
    try:
        import dataclasses
        if dataclasses.is_dataclass(cls):
            names |= {f.name for f in dataclasses.fields(cls)}
    except TypeError:  # pragma: no cover - defensive
        pass
    try:
        names |= set(inspect.signature(cls.__init__).parameters) - {"self"}
    except (TypeError, ValueError):  # pragma: no cover - C-level __init__
        pass
    return names


def first_supported_kwarg(cls, alternatives) -> tuple[str, object] | None:
    """First ``(name, value)`` from ``alternatives`` that ``cls`` accepts, else None.

    Transformers renames TrainingArguments knobs across major versions without
    changing what they mean: ``warmup_ratio`` -> ``warmup_steps`` (which takes a
    float in [0, 1) as a ratio), and ``group_by_length=True`` ->
    ``train_sampling_strategy="group_by_length"``. ACE has to run the SAME recipe on
    both the 4.57 platform-parity environment and the Transformers 5 environment
    Qwen3.5 needs, so each knob is expressed as an ordered list of equivalent
    spellings and the first one this build understands is used.

    Ordered oldest-first, so an environment exposing both keeps the historical
    spelling and the existing Qwen3/Phi-4/OLMo runs stay byte-reproducible.

    Deliberately returns at most ONE pair: passing both spellings at once would be
    a duplicate-argument error on any build that accepts both. Only the knobs handed
    in here are treated as version-dependent; every other TrainingArguments argument
    is passed directly, so a genuine configuration error still raises.
    """
    supported = supported_init_fields(cls)
    for name, value in alternatives:
        if name in supported:
            return name, value
    return None


def _load_bf16(cls, model_path: str, **extra):
    """from_pretrained in bfloat16, tolerant of the dtype/torch_dtype rename.

    Transformers renamed ``torch_dtype`` to ``dtype``. Rather than pin a version, try
    both and VERIFY the result: an unrecognised kwarg can be absorbed into the config
    instead of raising, which would silently give a float32 model — 4x the memory and
    a different numeric path. Checking ``model.dtype`` catches that regardless of
    which spelling the installed build honours.
    """
    import torch
    errors = []
    for key in ("dtype", "torch_dtype"):
        try:
            model = cls.from_pretrained(model_path, **{key: torch.bfloat16}, **extra)
        except TypeError as e:
            errors.append(f"{key}=: {type(e).__name__}: {e}")
            continue
        if model.dtype == torch.bfloat16:
            return model
        errors.append(f"{key}=: accepted but produced {model.dtype}")
        del model
        torch.cuda.empty_cache()
    raise RuntimeError(f"could not load {model_path} in bfloat16 -> " + " | ".join(errors))


def load_causal_lm(model_path: str, family: str | None = None, *, device_map="cuda",
                   verbose: bool = True) -> tuple[object, dict]:
    """Load the causal LM for training/scoring. Returns (model, provenance dict).

    Qwen3.5 takes the explicit text-only class and SDPA (see module docstring).
    Everything else keeps the historical AutoModelForCausalLM + flash-attention-2
    attempt with the SDPA fallback, unchanged.

    The provenance dict is recorded in the training config.json so an OOF file can
    always be traced to the exact class and attention kernel that produced it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if family is None:
        family, _ = detect_family(model_path)

    if family == QWEN35:
        # Explicitly the language model: Qwen3_5ForConditionalGeneration (what Auto*
        # may resolve for a multimodal checkpoint) would attach a vision tower ACE
        # neither feeds nor wants in the LoRA target set.
        from transformers import Qwen3_5ForCausalLM
        if verbose:
            print("[model] family=qwen3_5 -> Qwen3_5ForCausalLM, attn=sdpa, bf16")
        model = _load_bf16(Qwen3_5ForCausalLM, model_path,
                           attn_implementation="sdpa", device_map=device_map)
        return model, {"model_family": QWEN35,
                       "model_class": "Qwen3_5ForCausalLM",
                       "attn_implementation": "sdpa"}

    # ---- historical path: unchanged for Qwen3-4B/8B, Phi-4, OLMo3 --------------
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map=device_map)
        attn = "flash_attention_2"
    except (ImportError, ValueError) as e:
        if verbose:
            print(f"[model] flash-attn unavailable ({type(e).__name__}) -> falling back to "
                  f"SDPA (PyTorch fused kernels; ~10-20% slower at 8K ctx, same results)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map=device_map)
        attn = "sdpa"
    return model, {"model_family": GENERIC,
                   "model_class": type(model).__name__,
                   "attn_implementation": attn}
