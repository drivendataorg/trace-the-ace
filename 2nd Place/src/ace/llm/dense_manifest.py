#!/usr/bin/env python
"""Completion manifest for a dense Qwen3.5 export: written last, checked first.

A dense directory is not necessarily a finished export: if the tensor transplant fails, what
remains can be a complete, loadable copy of the unmodified base, which serves fine and is the
wrong model. The manifest is written only after the export has fully validated, so its presence
shows the transplant happened, and stage_llm.sh refuses any dense bundle without a valid one.
This is build-time tooling (not in build_zip.ACE_MODULES): inference never reads a manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "dense_export_manifest.json"

STATUS_COMPLETE = "complete"
FAMILY = "qwen3_5"
ADAPTER_MODE = "dense"

#: Fields every manifest must carry. Extra fields are allowed and encouraged.
REQUIRED_FIELDS = (
    "status", "family", "adapter_mode",
    "source_text_tensor_count", "mapped_tensor_count", "tied_alias_count",
    "unmapped_tensor_count", "shape_mismatch_count",
    "base_model", "adapter", "language_root",
)


class DenseManifestError(RuntimeError):
    """A dense checkpoint cannot be trusted: no manifest, or one that does not add up."""


def manifest_path(dense_dir) -> Path:
    return Path(dense_dir) / MANIFEST_NAME


def build_manifest(*, base_model: str, adapter: str, language_root: dict,
                   source_text_tensor_count: int, mapped_tensor_count: int,
                   tied_alias_count: int, replaced_tensor_count: int,
                   full_checkpoint_tensor_count: int, untouched_tensor_count: int,
                   architecture: str, extra: dict | None = None) -> dict:
    """Assemble the manifest for a successful export. Callers must validate first."""
    manifest = {
        "status": STATUS_COMPLETE,
        "family": FAMILY,
        "adapter_mode": ADAPTER_MODE,
        "architecture": architecture,
        "base_model": str(base_model),
        "adapter": str(adapter),
        "language_root": dict(language_root),
        "source_text_tensor_count": int(source_text_tensor_count),
        "mapped_tensor_count": int(mapped_tensor_count),
        "tied_alias_count": int(tied_alias_count),
        "replaced_tensor_count": int(replaced_tensor_count),
        "unmapped_tensor_count": 0,
        "shape_mismatch_count": 0,
        "full_checkpoint_tensor_count": int(full_checkpoint_tensor_count),
        "untouched_tensor_count": int(untouched_tensor_count),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(dense_dir, manifest: dict) -> Path:
    path = manifest_path(dense_dir)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(dense_dir) -> dict:
    path = manifest_path(dense_dir)
    if not path.is_file():
        raise DenseManifestError(
            f"no {MANIFEST_NAME} in {dense_dir}.\n\n"
            f"A dense bundle must be produced by scripts/export_qwen35_full_merged.py, "
            f"which writes that manifest only after the merged language weights have "
            f"been transplanted AND validated. A directory without it is most likely "
            f"the UNMODIFIED Qwen3.5 base left behind by a failed export — serving it "
            f"would test the base model, not the adapter.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise DenseManifestError(f"{path}: invalid JSON — {e}") from e
    if not isinstance(raw, dict):
        raise DenseManifestError(f"{path}: top level must be an object")
    return raw


def validate_manifest(manifest: dict, *, source: str = MANIFEST_NAME) -> dict:
    """Check a manifest is complete and internally consistent. Raises on any failure."""
    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise DenseManifestError(f"{source}: missing field(s) {missing}")

    if manifest["status"] != STATUS_COMPLETE:
        raise DenseManifestError(
            f"{source}: status={manifest['status']!r}, expected {STATUS_COMPLETE!r} — "
            f"this export did not finish")
    if manifest["family"] != FAMILY:
        raise DenseManifestError(f"{source}: family={manifest['family']!r}, "
                                 f"expected {FAMILY!r}")
    if manifest["adapter_mode"] != ADAPTER_MODE:
        raise DenseManifestError(f"{source}: adapter_mode={manifest['adapter_mode']!r}, "
                                 f"expected {ADAPTER_MODE!r}")

    ints = {}
    for field in ("source_text_tensor_count", "mapped_tensor_count", "tied_alias_count",
                  "unmapped_tensor_count", "shape_mismatch_count"):
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise DenseManifestError(f"{source}: {field}={value!r} must be an integer")
        ints[field] = value

    if ints["unmapped_tensor_count"] != 0:
        raise DenseManifestError(
            f"{source}: unmapped_tensor_count={ints['unmapped_tensor_count']} — the "
            f"export left language tensors carrying BASE weights")
    if ints["shape_mismatch_count"] != 0:
        raise DenseManifestError(
            f"{source}: shape_mismatch_count={ints['shape_mismatch_count']} — tensors "
            f"were transplanted between incompatible shapes")
    if ints["mapped_tensor_count"] <= 0:
        raise DenseManifestError(
            f"{source}: mapped_tensor_count={ints['mapped_tensor_count']} — nothing was "
            f"transplanted, so this is the unmodified base model")

    # The counts have to add up, or the manifest is describing a different export than
    # the one on disk: every source tensor is either mapped or a declared tied alias.
    expected = ints["mapped_tensor_count"] + ints["tied_alias_count"]
    if ints["source_text_tensor_count"] != expected:
        raise DenseManifestError(
            f"{source}: inconsistent counts — source_text_tensor_count="
            f"{ints['source_text_tensor_count']} but mapped({ints['mapped_tensor_count']})"
            f" + tied({ints['tied_alias_count']}) = {expected}")

    replaced = manifest.get("replaced_tensor_count")
    if replaced is not None and replaced != ints["mapped_tensor_count"]:
        raise DenseManifestError(
            f"{source}: replaced_tensor_count={replaced} != mapped_tensor_count="
            f"{ints['mapped_tensor_count']} — not every mapped tensor was written")

    if not manifest.get("language_root"):
        raise DenseManifestError(f"{source}: language_root is empty — the export did "
                                 f"not record where the language tower was written")
    return manifest


def require_valid_dense_export(dense_dir) -> dict:
    """Load + validate in one call. The staging guard's entry point."""
    return validate_manifest(load_manifest(dense_dir),
                             source=str(manifest_path(dense_dir)))