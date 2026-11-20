#!/usr/bin/env python
"""LLM bundle metadata: what engine to build, and whether the checkpoint can be served.

An ACE "bundle" is the deployable unit for one transcript reader — a `base/` model
directory plus zero or more `adapters/<name>/` LoRA directories, staged by
`scripts/stage_llm.sh` and scored by `ace.llm.score_worker`. Until E23 the bundle
carried no metadata at all: every reader was a Qwen3/Phi-4/OLMo checkpoint that vLLM
serves as a plain top-level causal LM with `enable_lora`, so one hard-coded `LLM(...)`
call covered them all.

Qwen3.5 breaks that assumption in a way that is worth stating precisely, because the
repository previously recorded it wrongly:

    vLLM 0.21 DOES support Qwen3.5. It registers `Qwen3_5ForConditionalGeneration`
    as a top-level multimodal generative architecture and exposes
    `language_model_only=True` to run just its language tower. What it does NOT
    register as a top-level text-generation architecture is `Qwen3_5ForCausalLM` —
    the class ACE trains against with PEFT. So a text-only merged checkpoint saved
    with `architectures=["Qwen3_5ForCausalLM"]` will be REJECTED by vLLM 0.21, while
    the full `Qwen3_5ForConditionalGeneration` checkpoint is served fine.

That single fact drives everything in this module: a Qwen3.5 bundle needs different
engine kwargs (`language_model_only=True`) and a hard pre-flight check on the base
checkpoint's top-level architecture, and neither may leak into the 4B/8B path.

DESIGN CONTRACT — backward compatibility is the whole point:

  * `bundle.json` is OPTIONAL. A bundle without one loads `LEGACY_META`, and
    `engine_kwargs()` then returns kwargs byte-identical to the pre-E23 hard-coded
    call. The shipped SUB-7/SUB-9 4B and 8B bundles are not regenerated and not
    touched. `tests/test_qwen35_deploy.py::test_legacy_*` pins this.
  * Nothing here imports vllm, torch or transformers, so the schema, the kwargs
    construction and the architecture validation are all unit-testable on CPU with
    none of them installed.
  * Architecture validation is DEPLOYMENT-ONLY. Training and `score_hf` deliberately
    keep using the text-only `Qwen3_5ForCausalLM`; this module never sees them.

TRAINING vs INFERENCE (do not conflate them):
    training   Transformers 5.15 + PEFT + flash-linear-attention 0.5.2, class
               `Qwen3_5ForCausalLM`
    inference  Transformers 4.57.6 + vLLM 0.21, architecture
               `Qwen3_5ForConditionalGeneration` + `language_model_only=True`,
               NO flash-linear-attention dependency (vLLM has its own GDN kernels)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ace.llm.model_family import GENERIC, QWEN35, _mentions_qwen35

# Engine policy constants. These reproduce the values score_worker.py hard-coded
# before E23; they are named here so the legacy path and the new path demonstrably
# share one source of truth rather than drifting apart.
GEN_MARGIN_TOKENS = 64          # max_model_len headroom over max_input_tokens
GPU_MEMORY_UTILIZATION = 0.90
MAX_LOGPROBS = 128
DEFAULT_MAX_LORA_RANK = 16

ADAPTER_MODE_LORA = "lora"
ADAPTER_MODE_DENSE = "dense"
ADAPTER_MODES = (ADAPTER_MODE_LORA, ADAPTER_MODE_DENSE)

BASE_MODE_EMBEDDED = "embedded"
BASE_MODE_EXTERNAL = "external"
BASE_MODES = (BASE_MODE_EMBEDDED, BASE_MODE_EXTERNAL)

# vLLM 0.21's registered top-level architecture for Qwen3.5.
QWEN35_VLLM_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
# The class ACE trains with PEFT. Valid for Transformers, NOT registered top-level in
# vLLM 0.21 — a deployment checkpoint declaring this will not load.
QWEN35_TRAINING_ARCHITECTURE = "Qwen3_5ForCausalLM"

BUNDLE_META_FILENAME = "bundle.json"


class BundleValidationError(RuntimeError):
    """A bundle cannot be deployed as declared. Raised before any engine is built."""


@dataclass(frozen=True)
class BundleMeta:
    """Deployment metadata for one reader bundle. Every field has a legacy-safe default.

    `folds` / `expected_folds` exist so a partially-trained reader cannot be mistaken
    for a complete cross-validated one. E23 is exactly that situation: fold 0 is
    scored while folds 1-4 are still training, and a bundle holding one adapter must
    announce itself as incomplete rather than passing as a five-fold reader.
    """

    engine: str = "vllm"
    architecture_family: str = GENERIC
    expected_architecture: str | None = None
    language_model_only: bool = False
    adapter_mode: str = ADAPTER_MODE_LORA
    base_mode: str = BASE_MODE_EMBEDDED
    external_base: str | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    max_lora_rank: int = DEFAULT_MAX_LORA_RANK
    folds: list = field(default_factory=list)
    expected_folds: list = field(default_factory=list)
    note: str = ""
    source: str = "default"          # "bundle.json" or "default" (provenance for logs)

    # ---- derived -------------------------------------------------------------
    @property
    def is_qwen35(self) -> bool:
        return self.architecture_family == QWEN35

    @property
    def uses_lora(self) -> bool:
        return self.adapter_mode == ADAPTER_MODE_LORA

    @property
    def incomplete_cv(self) -> bool:
        """True when the declared expected folds are not all present.

        No `expected_folds` means the question was never asked (every pre-E23
        bundle), which is reported as complete — those bundles ship a deliberate
        single full-fit adapter and there is nothing outstanding.
        """
        if not self.expected_folds:
            return False
        return sorted(map(str, self.folds)) != sorted(map(str, self.expected_folds))

    @property
    def missing_folds(self) -> list:
        have = set(map(str, self.folds))
        return [f for f in self.expected_folds if str(f) not in have]

    def describe(self) -> str:
        """One metadata-only line for deployment logs. Contains no test-data facts."""
        bits = [f"engine={self.engine}", f"family={self.architecture_family}",
                f"adapter_mode={self.adapter_mode}", f"base_mode={self.base_mode}"]
        if self.language_model_only:
            bits.append("language_model_only=True")
        if self.expected_folds:
            bits.append(f"folds={len(self.folds)}/{len(self.expected_folds)}"
                        + (" INCOMPLETE_CV" if self.incomplete_cv else ""))
        return " ".join(bits)


#: The metadata every bundle staged before E23 behaves as. Kept as a module constant
#: so tests can assert the legacy path is literally this object's behaviour.
LEGACY_META = BundleMeta()


def _require(condition, message: str) -> None:
    if not condition:
        raise BundleValidationError(message)


def parse_meta(raw: dict, *, source: str = BUNDLE_META_FILENAME) -> BundleMeta:
    """Validate a bundle.json dict into a BundleMeta.

    Unknown keys are REJECTED rather than ignored. A typo'd `language_model_only`
    that is silently dropped would build a Qwen3.5 engine in text-only mode and fail
    deep inside vLLM with an unrelated-looking error, or worse, load something that
    scores plausible but wrong numbers.
    """
    known = {f for f in BundleMeta.__dataclass_fields__ if f != "source"}
    # `incomplete_cv` may appear in a hand-written file as documentation; it is a
    # DERIVED property here, so accept it as an assertion and cross-check below.
    unknown = set(raw) - known - {"incomplete_cv", "name"}
    _require(not unknown, f"{source}: unknown key(s) {sorted(unknown)}; "
                          f"known keys are {sorted(known)}")

    family = str(raw.get("architecture_family", GENERIC))
    if _mentions_qwen35(family):
        family = QWEN35

    adapter_mode = str(raw.get("adapter_mode", ADAPTER_MODE_LORA))
    _require(adapter_mode in ADAPTER_MODES,
             f"{source}: adapter_mode={adapter_mode!r} not in {list(ADAPTER_MODES)}")
    base_mode = str(raw.get("base_mode", BASE_MODE_EMBEDDED))
    _require(base_mode in BASE_MODES,
             f"{source}: base_mode={base_mode!r} not in {list(BASE_MODES)}")
    engine = str(raw.get("engine", "vllm"))
    _require(engine == "vllm",
             f"{source}: engine={engine!r} unsupported; the submission runtime is vLLM")

    expected_arch = raw.get("expected_architecture")
    if expected_arch is None and family == QWEN35:
        # Default rather than require: the only architecture vLLM 0.21 registers for
        # this family is the conditional-generation wrapper.
        expected_arch = QWEN35_VLLM_ARCHITECTURE

    if base_mode == BASE_MODE_EXTERNAL:
        _require(raw.get("external_base"),
                 f"{source}: base_mode=external requires external_base "
                 f"(e.g. 'huggingface_models/Qwen/Qwen3.5-9B')")

    meta = BundleMeta(
        engine=engine,
        architecture_family=family,
        expected_architecture=expected_arch,
        language_model_only=bool(raw.get("language_model_only", False)),
        adapter_mode=adapter_mode,
        base_mode=base_mode,
        external_base=raw.get("external_base"),
        max_model_len=(int(raw["max_model_len"]) if raw.get("max_model_len") else None),
        dtype=(str(raw["dtype"]) if raw.get("dtype") else None),
        max_lora_rank=int(raw.get("max_lora_rank", DEFAULT_MAX_LORA_RANK)),
        folds=list(raw.get("folds", []) or []),
        expected_folds=list(raw.get("expected_folds", []) or []),
        note=str(raw.get("note", "")),
        source=source,
    )

    # A dense bundle carries its LoRA already merged in; asking for runtime LoRA too
    # would double-apply the deltas if an adapter dir were also present.
    if meta.adapter_mode == ADAPTER_MODE_DENSE:
        _require(not raw.get("max_lora_rank"),
                 f"{source}: adapter_mode=dense must not declare max_lora_rank")

    if "incomplete_cv" in raw:
        stated = bool(raw["incomplete_cv"])
        _require(stated == meta.incomplete_cv,
                 f"{source}: incomplete_cv={stated} contradicts folds={meta.folds} "
                 f"vs expected_folds={meta.expected_folds} "
                 f"(derived incomplete_cv={meta.incomplete_cv}); "
                 f"fix the fold lists rather than the flag")
    return meta


def load_bundle_meta(bundle_dir) -> BundleMeta:
    """Read `<bundle>/bundle.json`, or LEGACY_META when the file is absent.

    Absence is the normal case for every bundle staged before E23 and MUST keep
    behaving exactly as it did — that is the backward-compatibility contract.
    """
    path = Path(bundle_dir) / BUNDLE_META_FILENAME
    if not path.is_file():
        return BundleMeta(source="default")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BundleValidationError(f"{path}: invalid JSON — {e}") from e
    _require(isinstance(raw, dict), f"{path}: top level must be an object")
    return parse_meta(raw, source=str(path))


# ---------------------------------------------------------------------------
# base checkpoint validation
# ---------------------------------------------------------------------------
def read_base_architectures(base_dir) -> list:
    """Top-level `architectures` from a checkpoint's config.json (never nested)."""
    cfg_path = Path(base_dir) / "config.json"
    if not cfg_path.is_file():
        raise BundleValidationError(f"base checkpoint has no config.json: {cfg_path}")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise BundleValidationError(f"{cfg_path}: invalid JSON — {e}") from e
    return list(cfg.get("architectures") or [])


def validate_base_architecture(base_dir, meta: BundleMeta) -> list:
    """Pre-flight the base checkpoint against what vLLM 0.21 can actually serve.

    Only enforced for families that declare an `expected_architecture` — i.e. Qwen3.5.
    Generic bundles are not checked at all, so the 4B/8B/Phi-4/OLMo path is untouched.

    Returns the architecture list on success; raises BundleValidationError with an
    explanation on failure. Runs BEFORE the engine is constructed so the failure is a
    legible message rather than a vLLM registry traceback three minutes into a load.
    """
    if not meta.expected_architecture:
        return read_base_architectures(base_dir) if Path(base_dir).is_dir() else []

    archs = read_base_architectures(base_dir)
    if meta.expected_architecture in archs:
        return archs

    detail = (f"base checkpoint at {base_dir} declares architectures={archs!r}, "
              f"but this bundle requires {meta.expected_architecture!r}.")
    if any(a == QWEN35_TRAINING_ARCHITECTURE for a in archs):
        detail += (
            f"\n\n{QWEN35_TRAINING_ARCHITECTURE} is the TEXT-ONLY class ACE trains with "
            f"PEFT under Transformers 5. It is valid for training and for "
            f"ace.llm.score_hf, but vLLM 0.21 does NOT register it as a top-level "
            f"text-generation architecture — only the full "
            f"{QWEN35_VLLM_ARCHITECTURE} wrapper is registered (served with "
            f"language_model_only=True).\n"
            f"Fix: do not save a text-only merged checkpoint for deployment. Either ship "
            f"the full Qwen3.5 base + the LoRA adapter (adapter_mode=lora), or export a "
            f"dense checkpoint that keeps the full wrapper structure with "
            f"scripts/export_qwen35_full_merged.py (adapter_mode=dense).")
    elif any(_mentions_qwen35(a) for a in archs):
        detail += (f"\n\nThis looks like a Qwen3.5 checkpoint saved under an architecture "
                   f"vLLM 0.21 does not register. Deployment requires "
                   f"{QWEN35_VLLM_ARCHITECTURE}.")
    raise BundleValidationError(detail)


def resolve_base_path(bundle_dir, meta: BundleMeta, *, runtime_root=None) -> Path:
    """Where the engine should load the base model from.

    `base_mode=embedded` (every bundle to date) ships the weights inside the bundle.
    `base_mode=external` points at a model the competition runtime preloads under
    `huggingface_models/{org}/{model}/`, which keeps them out of the 60 GB zip — do
    NOT assume Qwen/Qwen3.5-9B is preloaded; that is why embedded stays the default.
    """
    bundle_dir = Path(bundle_dir)
    if meta.base_mode == BASE_MODE_EMBEDDED:
        return bundle_dir / "base"
    root = Path(runtime_root) if runtime_root is not None else Path.cwd()
    external = Path(meta.external_base)
    path = external if external.is_absolute() else root / external
    if not path.is_dir():
        raise BundleValidationError(
            f"base_mode=external but the preloaded model is absent: {path}. "
            f"Stage the bundle with base_mode=embedded, or confirm the runtime "
            f"provides {meta.external_base}.")
    return path


def base_is_available(bundle_dir, meta: BundleMeta | None = None, *,
                      runtime_root=None) -> bool:
    """Can this bundle's base model be loaded here? Mode-aware, not path-guessing.

    Callers used to ask `(<bundle>/base).exists()`, which is right for an embedded
    bundle and WRONG for an external one — an external bundle deliberately ships no
    local `base/`, so that test silently classified a perfectly good reader as "not
    staged". This is the mode-aware replacement.

    Malformed metadata is NOT swallowed into a False: a bundle.json that cannot be
    parsed is a deployment error and must be raised, never quietly turned into a
    disabled reader.
    """
    meta = meta if meta is not None else load_bundle_meta(bundle_dir)
    if meta.base_mode == BASE_MODE_EMBEDDED:
        return (Path(bundle_dir) / "base").exists()
    try:
        resolve_base_path(bundle_dir, meta, runtime_root=runtime_root)
    except BundleValidationError:
        return False
    return True


def describe_base_requirement(bundle_dir, meta: BundleMeta) -> str:
    """Why a bundle's base is unavailable, in the terms of its own mode."""
    if meta.base_mode == BASE_MODE_EMBEDDED:
        return f"{bundle_dir}: base_mode=embedded but {Path(bundle_dir) / 'base'} is absent"
    return (f"{bundle_dir}: base_mode=external but the runtime does not provide "
            f"{meta.external_base!r}")


# ---------------------------------------------------------------------------
# engine construction
# ---------------------------------------------------------------------------
def engine_kwargs(meta: BundleMeta, *, model_path, max_input_tokens: int,
                  enforce_eager: bool) -> dict:
    """Exact kwargs for `vllm.LLM(**kwargs)` for this bundle.

    LEGACY GUARANTEE: with `LEGACY_META` this returns precisely the pre-E23
    hard-coded call — same keys, same values, no `dtype`, no `language_model_only`.
    A Qwen3 bundle must not acquire new engine arguments as a side effect of E23,
    because "equivalent kwargs" is the only thing standing between this refactor and
    a silent change to the submission of record.

    `max_model_len` defaults to `max_input_tokens + GEN_MARGIN_TOKENS`, matching the
    existing policy: prompts are budgeted to exactly `max_input_tokens`, and the
    generation prompt needs headroom on top. Setting `max_model_len: 8192` in
    bundle.json for an 8192-token prompt budget would leave no margin — only override
    it if you have measured that the prompts fit.
    """
    kwargs = {
        "model": str(model_path),
        "max_model_len": int(meta.max_model_len or (max_input_tokens + GEN_MARGIN_TOKENS)),
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enable_prefix_caching": True,
        "max_logprobs": MAX_LOGPROBS,
        "disable_log_stats": True,
        "enforce_eager": bool(enforce_eager),
    }
    if meta.uses_lora:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = int(meta.max_lora_rank)
    else:
        kwargs["enable_lora"] = False
    if meta.language_model_only:
        # Qwen3.5 only: serve the language tower of the registered multimodal
        # architecture. Never sent for a generic bundle — vLLM 0.21 rejects the
        # argument for architectures that are not multimodal.
        kwargs["language_model_only"] = True
    if meta.dtype:
        kwargs["dtype"] = meta.dtype
    # Key order is normalised so a kwargs diff in tests reads cleanly.
    legacy_order = ["model", "enable_lora", "max_lora_rank", "max_model_len",
                    "gpu_memory_utilization", "enable_prefix_caching", "max_logprobs",
                    "disable_log_stats", "enforce_eager", "language_model_only", "dtype"]
    return {k: kwargs[k] for k in legacy_order if k in kwargs}


def check_engine_supports(meta: BundleMeta, *, verbose: bool = True):
    """Verify the environment can actually run THIS bundle, and report what it is.

    CAPABILITY check, not a version pin. The competition runtime may legitimately move
    to a newer vLLM/Transformers/CUDA/GPU; production must keep working there, so
    nothing here compares version strings for the purpose of blocking. What is checked
    is whether the arguments this bundle needs exist in the installed build —
    `language_model_only` for Qwen3.5, the LoRA arguments for adapter_mode=lora — plus
    the basics (CUDA present, vLLM importable). Those are real incompatibilities.

    A runtime that differs from the parity-certified reference produces a WARNING and
    continues. See ace.llm.runtime_check for why the two are kept apart.
    """
    from ace.llm.runtime_check import inspect_runtime_environment, require_runtime

    report = inspect_runtime_environment(architecture_family=meta.architecture_family,
                                         adapter_mode=meta.adapter_mode)
    if verbose:
        # Metadata only — nothing derived from the test data.
        for line in report.format_lines():
            print(f"[worker] {line}", flush=True)
    require_runtime(report, context="LLM bundle")
    return report


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------
def adapter_dirs(bundle_dir) -> list:
    """Sorted adapter directories, or [] for a dense bundle with no adapters/."""
    adapters = Path(bundle_dir) / "adapters"
    if not adapters.is_dir():
        return []
    return sorted((p for p in adapters.iterdir() if p.is_dir()), key=lambda p: p.name)


def fold_label(adapter_name: str):
    """`fold3` -> 3, `full` -> 'full', anything else -> the name.

    Mirrors the stage_llm.sh / e13_qwen.sh naming convention so fold coverage can be
    reported without a second source of truth for what an adapter directory is called.
    """
    if adapter_name.startswith("fold") and adapter_name[4:].isdigit():
        return int(adapter_name[4:])
    return adapter_name


def validate_bundle(bundle_dir, meta: BundleMeta | None = None, *,
                    runtime_root=None, resolve_base: bool = True) -> BundleMeta:
    """Full structural pre-flight for a bundle directory.

    Checks only what is cheap and catches real deployment failures: the declared
    adapter mode against what is on disk, and (Qwen3.5 only) the base architecture —
    of the RESOLVED base, so an external/preloaded checkpoint is validated exactly
    like an embedded one rather than skipped.

    `resolve_base=False` skips base resolution entirely. That is for BUILD time: an
    external bundle names a model the *runtime* provides, which by definition cannot be
    resolved on the machine assembling the zip. Metadata is still fully validated there
    (`parse_meta` already requires `external_base` for `base_mode=external`), and the
    architecture check then happens in the worker, where the path exists.
    """
    bundle_dir = Path(bundle_dir)
    meta = meta if meta is not None else load_bundle_meta(bundle_dir)
    adapters = adapter_dirs(bundle_dir)

    if meta.uses_lora:
        _require(adapters, f"{bundle_dir}: adapter_mode=lora but no adapters/*/ present")
    else:
        _require(not adapters,
                 f"{bundle_dir}: adapter_mode=dense but adapters/ contains "
                 f"{[p.name for p in adapters]} — a dense checkpoint already has the "
                 f"LoRA deltas merged in; serving it with a runtime adapter as well "
                 f"would apply them twice")

    # LoRA mode only: `folds` names the adapters that are served, so it must match
    # what is on disk. For a DENSE bundle `folds` records which fold's adapter was
    # MERGED into the checkpoint — there is nothing on disk to compare it against, and
    # the merge itself is validated by export_qwen35_full_merged.py.
    if meta.folds and meta.uses_lora:
        have = {str(fold_label(p.name)) for p in adapters}
        declared = {str(f) for f in meta.folds}
        _require(declared == have,
                 f"{bundle_dir}: bundle.json declares folds={sorted(declared)} but "
                 f"adapters/ holds {sorted(have)}")

    if resolve_base:
        # One code path for both modes: resolve first, then validate whatever the
        # resolution produced. An external Qwen3.5 checkpoint declaring the text-only
        # Qwen3_5ForCausalLM must be rejected just as loudly as an embedded one.
        validate_base_architecture(
            resolve_base_path(bundle_dir, meta, runtime_root=runtime_root), meta)
    return meta