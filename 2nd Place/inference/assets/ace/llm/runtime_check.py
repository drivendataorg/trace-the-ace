#!/usr/bin/env python
"""Runtime inspection: CAN this environment run the deployment, and IS it the one we certified?

Those are two different questions and conflating them is a production hazard.

    runtime_compatible       the capabilities this deployment actually needs are
                             present. If False, inference cannot work — BLOCK.
    reference_runtime_match  the installed versions equal the environment in which
                             HF-vLLM numerical parity was measured. If False, the
                             parity evidence is one step less direct — WARN.

**Production inference must never fail merely because versions differ.** The
competition runtime may legitimately move to vLLM 0.22, a newer Transformers, CUDA 13
or a better GPU; a submission that asserts `transformers == "4.57.6"` would break on a
runtime that is perfectly capable of serving it. So nothing here pins a version for
production. Capabilities are probed directly — does `EngineArgs` accept
`language_model_only`, does `torch.cuda.is_available()` — because an API signature is
the fact that matters, and a version string is only a proxy for it.

Reference versions are PROVENANCE. `scripts/e23_qwen35_deploy_check.py` may opt into
enforcing them with `--require-reference-runtime`, which is a *certification* mode for
deliberately reproducing the parity environment. That flag exists for development and
must never gate production.

Everything is injectable through `probe=`, so every branch is unit-testable on a CPU
box with neither torch nor vLLM installed.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

#: The environment in which HF-vLLM numerical parity was (or will be) certified.
#: PROVENANCE ONLY — never a production constraint. See the module docstring.
REFERENCE_RUNTIME = {
    "python": "3.12",
    "transformers": "4.57.6",
    "torch": "2.11.0+cu129",
    "vllm": "0.21.0+cu129",
    "cuda": "12.9",
}

#: Minimum Python the competition runtime is built on. Below this, ACE's own code is
#: not expected to run; above it is fine and merely worth noting.
MIN_PYTHON = (3, 12)


@dataclass
class RuntimeReport:
    python_version: str = ""
    torch_version: str | None = None
    transformers_version: str | None = None
    vllm_version: str | None = None
    cuda_runtime: str | None = None
    cuda_available: bool = False
    gpu_count: int = 0
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    capabilities: dict = field(default_factory=dict)
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def runtime_compatible(self) -> bool:
        """True when nothing this deployment REQUIRES is missing."""
        return not self.blockers

    @property
    def reference_runtime_match(self) -> bool:
        """True when the installed versions equal the certified parity environment."""
        if self.python_version.rsplit(".", 1)[0] != REFERENCE_RUNTIME["python"]:
            return False
        for key, actual in (("transformers", self.transformers_version),
                            ("torch", self.torch_version),
                            ("vllm", self.vllm_version),
                            ("cuda", self.cuda_runtime)):
            if actual != REFERENCE_RUNTIME[key]:
                return False
        return True

    def versions(self) -> dict:
        return {"python": self.python_version, "torch": self.torch_version,
                "transformers": self.transformers_version, "vllm": self.vllm_version,
                "cuda": self.cuda_runtime}

    def format_lines(self) -> list:
        """Metadata-only report lines. Nothing here derives from the test data."""
        lines = [
            "  actual runtime:",
            f"    python       {self.python_version}",
            f"    torch        {self.torch_version or 'ABSENT'}",
            f"    transformers {self.transformers_version or 'ABSENT'}",
            f"    vllm         {self.vllm_version or 'ABSENT'}",
            f"    cuda         {self.cuda_runtime or 'none'}"
            + (f"   gpu {self.gpu_name}" if self.gpu_name else "")
            + (f" ({self.gpu_memory_gb:.0f} GB)" if self.gpu_memory_gb else ""),
            "  reference (parity-certified) runtime:",
        ]
        lines += [f"    {k:<12} {v}" for k, v in REFERENCE_RUNTIME.items()]
        lines.append(f"  runtime compatibility  : "
                     f"{'PASS' if self.runtime_compatible else 'FAIL'}")
        lines.append(f"  reference runtime match: "
                     f"{'YES' if self.reference_runtime_match else 'NO'}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for b in self.blockers:
            lines.append(f"  BLOCKER: {b}")
        return lines


def _version_of(name: str):
    try:
        mod = __import__(name)
    except Exception:  # noqa: BLE001 - a broken install is "absent" for our purposes
        return None
    return getattr(mod, "__version__", "unknown")


def _default_probe() -> dict:
    """Read the real environment. Every lookup is defensive: a missing or broken
    optional dependency must produce a report, not an exception."""
    facts: dict = {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "transformers_version": _version_of("transformers"),
        "vllm_version": _version_of("vllm"),
        "torch_version": None,
        "cuda_runtime": None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "engine_args_fields": None,
    }
    try:
        import torch
        facts["torch_version"] = torch.__version__
        facts["cuda_runtime"] = torch.version.cuda
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if facts["cuda_available"]:
            facts["gpu_count"] = int(torch.cuda.device_count())
            if facts["gpu_count"]:
                props = torch.cuda.get_device_properties(0)
                facts["gpu_name"] = props.name
                facts["gpu_memory_gb"] = props.total_memory / 1e9
    except Exception:  # noqa: BLE001
        pass
    # Signature inspection beats version-string guessing: what matters is whether THIS
    # build accepts the arguments the bundle needs, whatever it calls itself.
    try:
        from vllm import EngineArgs

        from ace.llm.model_family import supported_init_fields
        facts["engine_args_fields"] = supported_init_fields(EngineArgs)
    except Exception:  # noqa: BLE001
        pass
    return facts


def inspect_runtime_environment(*, architecture_family: str = "generic",
                                adapter_mode: str = "lora",
                                probe: dict | None = None) -> RuntimeReport:
    """Inspect the environment and decide compatibility for THIS deployment shape.

    `architecture_family` / `adapter_mode` matter because the requirements differ:
    a Qwen3.5 bundle needs `language_model_only`; only a LoRA bundle needs the LoRA
    arguments. A dense Qwen3.5 bundle does not require live-LoRA support at all.
    """
    facts = dict(probe) if probe is not None else _default_probe()
    r = RuntimeReport(
        python_version=facts.get("python_version", ""),
        torch_version=facts.get("torch_version"),
        transformers_version=facts.get("transformers_version"),
        vllm_version=facts.get("vllm_version"),
        cuda_runtime=facts.get("cuda_runtime"),
        cuda_available=bool(facts.get("cuda_available")),
        gpu_count=int(facts.get("gpu_count") or 0),
        gpu_name=facts.get("gpu_name"),
        gpu_memory_gb=facts.get("gpu_memory_gb"),
    )

    # ---- capabilities that genuinely gate inference ---------------------------
    try:
        parts = tuple(int(x) for x in r.python_version.split(".")[:2])
    except ValueError:
        parts = ()
    r.capabilities["python_ok"] = bool(parts and parts >= MIN_PYTHON)
    if not r.capabilities["python_ok"]:
        r.blockers.append(f"python {r.python_version or '?'} is below the "
                          f"{'.'.join(map(str, MIN_PYTHON))} the runtime is built on")
    elif parts[:2] != tuple(int(x) for x in REFERENCE_RUNTIME["python"].split(".")):
        r.warnings.append(f"python {r.python_version} differs from the reference "
                          f"{REFERENCE_RUNTIME['python']}")

    r.capabilities["torch"] = r.torch_version is not None
    if not r.capabilities["torch"]:
        r.blockers.append("torch is not importable")
    r.capabilities["transformers"] = r.transformers_version is not None
    if not r.capabilities["transformers"]:
        r.blockers.append("transformers is not importable")
    r.capabilities["vllm"] = r.vllm_version is not None
    if not r.capabilities["vllm"]:
        r.blockers.append("vllm is not importable — the submission serves through vLLM")

    r.capabilities["cuda_available"] = r.cuda_available
    if not r.cuda_available:
        r.blockers.append("no CUDA device available — vLLM cannot serve these readers")
    r.capabilities["gpu_present"] = r.gpu_count >= 1
    if r.cuda_available and r.gpu_count < 1:
        r.blockers.append("CUDA reports available but no GPU was enumerated")

    # ---- engine-argument capabilities, by signature ---------------------------
    fields = facts.get("engine_args_fields")
    known = fields is not None
    r.capabilities["engine_args_introspectable"] = known
    if r.capabilities["vllm"] and not known:
        # Advisory only. Failing here would block a future vLLM that reorganised
        # EngineArgs but still supports everything we need.
        r.warnings.append("could not introspect vLLM EngineArgs; engine-argument "
                          "support will be discovered when the engine is built")

    needs_lmo = architecture_family == "qwen3_5"
    needs_lora = adapter_mode == "lora"
    if needs_lmo:
        ok = (None if not known else "language_model_only" in fields)
        r.capabilities["language_model_only"] = ok
        if ok is False:
            r.blockers.append(
                "the installed vLLM does not accept language_model_only, which every "
                "Qwen3.5 bundle requires (it is served as the language tower of "
                "Qwen3_5ForConditionalGeneration)")
    if needs_lora:
        ok = (None if not known
              else {"enable_lora", "max_lora_rank"}.issubset(fields))
        r.capabilities["lora_args"] = ok
        if ok is False:
            r.blockers.append("the installed vLLM does not accept enable_lora / "
                              "max_lora_rank, which this adapter_mode=lora bundle needs")

    if not r.reference_runtime_match and r.runtime_compatible:
        r.warnings.append(
            "runtime differs from the environment used for HF-vLLM parity validation. "
            "Required deployment capabilities are available, so inference will "
            "continue; the parity evidence is one step less direct.")
    return r


def require_runtime(report: RuntimeReport, *, context: str = "deployment") -> None:
    """Raise on ACTUAL incompatibility. Never on a mere version difference."""
    if report.runtime_compatible:
        return
    raise RuntimeError(
        f"{context}: this environment cannot run the bundle — "
        + "; ".join(report.blockers)
        + ". (Version differences alone are never a blocker; these are missing "
          "capabilities.)")