#!/usr/bin/env python
r"""Merge the Qwen3.5 LoRA into a dense checkpoint that vLLM 0.21 can serve (training stage 9).

Training uses the text-only Qwen3_5ForCausalLM, which vLLM 0.21 does not register; it serves
Qwen3_5ForConditionalGeneration with language_model_only=True. The merged language weights are
therefore transplanted into a copy of the full checkpoint, leaving its config, processor files,
vision tower (model.visual.*) and multi-token-prediction head (mtp.*) untouched. One language
root is discovered for the whole state dict instead of matching keys by suffix, because
mtp.layers.0.* and several norms share suffixes with language weights. Work happens in
<out>.tmp and the completion manifest is written last, so a failed run never leaves something
that looks like a finished export. CPU only; peak memory about one bf16 9B model plus a shard.

    python scripts/export_qwen35_full_merged.py --base Qwen/Qwen3.5-9B \
        --adapter outputs/models/e23_qwen35_9b/full/adapter --out outputs/models/e23_qwen35_9b/full/dense
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ace.llm.bundle import (QWEN35_TRAINING_ARCHITECTURE,  # noqa: E402
                            QWEN35_VLLM_ARCHITECTURE, read_base_architectures)
from ace.llm.dense_manifest import (MANIFEST_NAME, build_manifest,  # noqa: E402
                                    validate_manifest, write_manifest)

# Files that must survive into the export for vLLM/Transformers to load it offline.
TOKENIZER_PATTERNS = ("tokenizer*", "vocab*", "merges*", "special_tokens*",
                      "added_tokens*", "*.jinja", "chat_template*")

#: Tensors allowed to be unmapped without being a discovered tied alias. Empty on purpose:
#: a hole in language-weight coverage leaves base weights in the dense export, a silently
#: wrong model. Tied parameters are handled by `discover_tied_aliases`, which asks the loaded
#: model. Add an entry only with a comment saying which checkpoint needs it and why.
STATIC_ALIAS_ALLOWLIST: frozenset = frozenset()


class ExportError(RuntimeError):
    """The export cannot proceed. Raised before anything irreversible happens."""


# ---------------------------------------------------------------------------
# language-root discovery + explicit mapping (pure functions)
# ---------------------------------------------------------------------------
def _split_head(key: str) -> tuple[str, str]:
    head, _, rest = key.partition(".")
    return head, rest


def discover_language_root(text_keys, full_keys) -> dict:
    """Find the single insertion per top-level head that maps the text model into the full
    checkpoint's language tower.

    The rewrite is head.rest -> head.INSERT.rest (INSERT="" when the name is identical). A
    candidate is accepted only if it maps every text key under that head, so mtp (11 layer
    tensors, not 426) and the vision tower cannot win. Returns e.g.
    {"model": "language_model", "lm_head": ""}; raises ExportError if a head has no fully
    covering candidate, or more than one.
    """
    full = set(full_keys)
    by_head = defaultdict(list)
    for k in text_keys:
        head, rest = _split_head(k)
        if not rest:
            raise ExportError(f"cannot map single-component tensor name {k!r}")
        by_head[head].append(rest)

    roots: dict = {}
    for head, rests in sorted(by_head.items()):
        # Candidate insertions are read off the checkpoint, using the longest source name
        # as the anchor: the more specific the anchor, the fewer false candidates.
        anchor = max(rests, key=len)
        prefix, suffix = head + ".", "." + anchor
        candidates = {""} if f"{head}.{anchor}" in full else set()
        for f in full:
            if f.startswith(prefix) and f.endswith(suffix):
                middle = f[len(prefix):len(f) - len(suffix)]
                if middle:
                    candidates.add(middle)

        covering = []
        for cand in sorted(candidates):
            joiner = f"{head}.{cand}." if cand else f"{head}."
            if all(f"{joiner}{r}" in full for r in rests):
                covering.append(cand)

        if not covering:
            best, best_n = None, -1
            for cand in sorted(candidates):
                joiner = f"{head}.{cand}." if cand else f"{head}."
                n = sum(1 for r in rests if f"{joiner}{r}" in full)
                if n > best_n:
                    best, best_n = cand, n
            raise ExportError(
                f"no insertion under {head!r} maps all {len(rests)} source tensors into "
                f"the full checkpoint (best candidate {best!r} covered {best_n}). The "
                f"checkpoints do not correspond, or this Transformers version nests the "
                f"language tower differently.")
        if len(covering) > 1:
            raise ExportError(
                f"ambiguous language root under {head!r}: {covering!r} each map all "
                f"{len(rests)} source tensors. Refusing to guess which is the language "
                f"tower.")
        roots[head] = covering[0]
    return roots


def build_explicit_mapping(text_keys, full_keys, roots: dict | None = None) -> dict:
    """{text_key: full_key} through the discovered root; deterministic.

    Each destination must exist in the checkpoint, and no two source tensors may share one: a
    collision would silently overwrite one with the other.
    """
    full = set(full_keys)
    roots = roots if roots is not None else discover_language_root(text_keys, full_keys)

    mapping, absent = {}, []
    for k in text_keys:
        head, rest = _split_head(k)
        insert = roots.get(head)
        if insert is None:
            absent.append(k)
            continue
        dest = f"{head}.{insert}.{rest}" if insert else f"{head}.{rest}"
        if dest not in full:
            absent.append(k)
            continue
        mapping[k] = dest

    if absent:
        raise ExportError(
            f"{len(absent)} source tensor(s) have no destination under the discovered "
            f"root {roots!r}:\n" + format_keys(absent))
    dupes = [d for d, n in Counter(mapping.values()).items() if n > 1]
    if dupes:
        raise ExportError(f"mapping is not injective — {len(dupes)} destination(s) "
                          f"claimed twice:\n" + format_keys(dupes))
    return mapping


def discover_tied_aliases(model) -> set:
    """Tensor names that legitimately need no replacement, read from the loaded model.

    Tied parameters (classically lm_head.weight tied to the input embedding) appear in the text
    model's state_dict() but may have no separate counterpart in the full checkpoint. They come
    from _tied_weights_keys and, when config.tie_word_embeddings is set, the output embedding's
    parameter name; nothing is matched by substring. Qwen3.5-9B does not tie them, so this is
    empty there and lm_head.weight is transplanted like any other tensor.
    """
    aliases = set(STATIC_ALIAS_ALLOWLIST)
    for key in (getattr(model, "_tied_weights_keys", None) or []):
        aliases.add(str(key))
    cfg = getattr(model, "config", None)
    if getattr(cfg, "tie_word_embeddings", False):
        getter = getattr(model, "get_output_embeddings", None)
        out = getter() if callable(getter) else None
        if out is not None:
            for name, module in model.named_modules():
                if module is out:
                    aliases.add(f"{name}.weight" if name else "weight")
    return aliases


def analyse_coverage(text_keys, full_keys, mapping, *, tied_aliases=(),
                     roots: dict | None = None) -> dict:
    """Account for every language tensor (a pure function).

    A count of mapped tensors does not show whether the unmapped ones were allowed to be: a hole
    leaves base-model weights in the dense export, which loads cleanly and scores wrongly.
    """
    text_keys, tied = list(text_keys), set(tied_aliases)
    full_set = set(full_keys)
    mapped = {k: v for k, v in mapping.items() if k in set(text_keys)}

    unmapped = [k for k in text_keys if k not in mapped]
    unmapped_tied = sorted(k for k in unmapped if k in tied)
    unmapped_unexpected = sorted(k for k in unmapped if k not in tied)

    destinations = sorted(set(mapped.values()))
    absent_destinations = sorted(d for d in destinations if d not in full_set)

    # Every destination must sit under the discovered root for its head, which keeps
    # `model.visual.*` and `mtp.*` out of the target set by construction.
    unexpected_destinations = []
    if roots:
        allowed_prefixes = {f"{h}.{i}." if i else f"{h}." for h, i in roots.items()}
        unexpected_destinations = sorted(
            d for d in destinations
            if not any(d.startswith(p) for p in allowed_prefixes))

    return {"n_source": len(text_keys), "n_mapped": len(mapped),
            "mapped": mapped, "destinations": destinations,
            "roots": dict(roots or {}),
            "unmapped_tied": unmapped_tied,
            "unmapped_unexpected": unmapped_unexpected,
            "absent_destinations": absent_destinations,
            "unexpected_destinations": unexpected_destinations,
            "complete": not (unmapped_unexpected or absent_destinations
                             or unexpected_destinations)}


def format_keys(keys, limit: int = 8) -> str:
    keys = list(keys)
    head = "\n".join(f"      {k}" for k in keys[:limit])
    if len(keys) > limit:
        head += f"\n      ... and {len(keys) - limit} more (total {len(keys)})"
    return head


# ---------------------------------------------------------------------------
# model loading / shard rewriting
# ---------------------------------------------------------------------------
def resolve_local(model_id_or_path: str) -> Path:
    """Local directory for a base model, downloading the snapshot if given a hub id."""
    p = Path(model_id_or_path).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    print(f"[base] resolving hub id {model_id_or_path} ...")
    return Path(snapshot_download(model_id_or_path))


def load_merged_text_model(base_dir: Path, adapter: Path):
    """Text-only model with the LoRA merged in, on CPU in bf16."""
    import torch
    from peft import PeftModel
    from transformers import Qwen3_5ForCausalLM

    print(f"[merge] loading {QWEN35_TRAINING_ARCHITECTURE} on CPU (bf16) ...")
    # Same class and dtype as training and score_hf: the merged weights must be the ones
    # that produced the reference OOF predictions.
    try:
        model = Qwen3_5ForCausalLM.from_pretrained(
            str(base_dir), dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True)
    except TypeError:                                   # Transformers <5 spelling
        model = Qwen3_5ForCausalLM.from_pretrained(
            str(base_dir), torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True)
    if model.dtype != torch.bfloat16:
        raise ExportError(f"loaded as {model.dtype}, expected bfloat16")
    print(f"[merge] attaching adapter {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter), torch_device="cpu")
    print("[merge] merge_and_unload() ...")
    merged = model.merge_and_unload()
    merged.eval()
    return merged


def read_full_keys(directory: Path) -> list:
    from safetensors import safe_open
    keys = []
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            keys.extend(f.keys())
    return keys


def rewrite_shards(out_dir: Path, merged_sd: dict, mapping: dict) -> tuple:
    """Replace mapped tensors in the copied checkpoint's shards. Returns (written, bad).

    `written` is the set of destination keys, so the caller can prove every mapped destination was
    reached (a count cannot tell the same number of keys from the same keys). All shape
    mismatches are collected, so one run reports the whole problem.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    shards = sorted(out_dir.glob("*.safetensors"))
    if not shards:
        raise ExportError(f"no *.safetensors in {out_dir}")
    full_to_text = {v: k for k, v in mapping.items()}

    # Check shapes across all shards before writing anything, so a mismatch in the last
    # shard cannot leave the first ones half-transplanted.
    mismatches = []
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k in full_to_text:
                    dst = tuple(f.get_slice(k).get_shape())
                    src = tuple(merged_sd[full_to_text[k]].shape)
                    if src != dst:
                        mismatches.append((full_to_text[k], k, src, dst))
    if mismatches:
        return set(), mismatches

    written: set = set()
    for shard in shards:
        tensors = load_file(str(shard))
        touched = [k for k in tensors if k in full_to_text]
        if not touched:
            print(f"[write] {shard.name}: unchanged ({len(tensors)} tensors)")
            del tensors
            continue
        for k in touched:
            src = merged_sd[full_to_text[k]]
            tensors[k] = src.detach().to(tensors[k].dtype).contiguous().clone()
        # Write a new file and swap, never in place. `load_file` memory-maps the shard,
        # and overwriting a path whose mapping is still open fails outright on Windows
        # (os error 1224) and is unsafe anywhere. The swap also makes each shard write
        # atomic: an interruption leaves the original shard intact.
        # Preserve the metadata header; safetensors requires format=pt.
        staged = shard.with_name(shard.name + ".new")
        save_file(tensors, str(staged), metadata={"format": "pt"})
        n_total = len(tensors)
        del tensors                       # drop every reference to the mapping
        gc.collect()
        os.replace(staged, shard)
        written.update(touched)
        print(f"[write] {shard.name}: replaced {len(touched)}/{n_total} tensors")
    return written, []


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def validate_export(out_dir: Path, coverage: dict, replaced: set,
                    mismatches=()) -> None:
    """Structural checks; each one catches a real way the export can go wrong.

    Structural validity is not inference parity: it shows the checkpoint has the shape vLLM 0.21
    needs and every language weight carries the merged LoRA. Whether it computes the same numbers
    as score_hf is settled on a GPU by training stage 9 (scripts/stage_llm.sh), then
    scripts/run_inference.sh.
    """
    print("\n=== structural validation ===")
    problems = []

    if mismatches:
        problems.append(
            f"{len(mismatches)} tensor(s) have mismatched shapes:\n"
            + format_keys(f"{t} -> {f}: source {s} vs destination {d}"
                          for t, f, s, d in mismatches))

    mapped_dest = set(coverage["destinations"])
    if coverage["unmapped_unexpected"]:
        problems.append(
            f"{len(coverage['unmapped_unexpected'])} language tensor(s) were never "
            f"mapped into the full checkpoint, and are NOT tied aliases. The export "
            f"would keep BASE weights for them:\n"
            + format_keys(coverage["unmapped_unexpected"]))
    if coverage["absent_destinations"]:
        problems.append(
            f"{len(coverage['absent_destinations'])} destination key(s) do not exist in "
            f"the checkpoint:\n" + format_keys(coverage["absent_destinations"]))
    if coverage["unexpected_destinations"]:
        problems.append(
            f"{len(coverage['unexpected_destinations'])} tensor(s) mapped OUTSIDE the "
            f"discovered language root {coverage.get('roots')}:\n"
            + format_keys(coverage["unexpected_destinations"]))
    missing_writes = sorted(mapped_dest - set(replaced))
    if missing_writes:
        problems.append(
            f"{len(missing_writes)} mapped tensor(s) were never written to a shard:\n"
            + format_keys(missing_writes))
    stray_writes = sorted(set(replaced) - mapped_dest)
    if stray_writes:
        problems.append(f"{len(stray_writes)} tensor(s) were written but not mapped:\n"
                        + format_keys(stray_writes))
    if not problems:
        print(f"[ok] language coverage complete: {coverage['n_mapped']}/"
              f"{coverage['n_source']} source tensors mapped and written"
              + (f", {len(coverage['unmapped_tied'])} tied alias(es) skipped "
                 f"({', '.join(coverage['unmapped_tied'])})"
                 if coverage["unmapped_tied"] else ""))

    archs = read_base_architectures(out_dir)
    if QWEN35_VLLM_ARCHITECTURE in archs:
        print(f"[ok] top-level architectures = {archs}")
    else:
        problems.append(f"architectures={archs} does not contain "
                        f"{QWEN35_VLLM_ARCHITECTURE} — vLLM 0.21 will reject this")

    leftovers = [p.name for p in out_dir.iterdir()
                 if p.name.startswith("adapter_") or p.name == "adapter"]
    if leftovers:
        problems.append(f"PEFT adapter artifacts left in the export: {leftovers}")
    else:
        print("[ok] no PEFT adapter modules remain")

    have_tok = [p.name for pat in TOKENIZER_PATTERNS for p in out_dir.glob(pat)]
    if any(n.startswith("tokenizer") for n in have_tok):
        print(f"[ok] tokenizer files present ({len(have_tok)})")
    else:
        problems.append("no tokenizer* files in the export — offline load will fail")

    index = out_dir / "model.safetensors.index.json"
    if index.is_file():
        idx = json.loads(index.read_text(encoding="utf-8"))
        weight_map = idx.get("weight_map", {})
        missing = sorted({f for f in weight_map.values() if not (out_dir / f).is_file()})
        if missing:
            problems.append(f"index references absent shards: {missing[:3]}")
        else:
            print(f"[ok] safetensors index valid ({len(weight_map)} tensors, "
                  f"{len(set(weight_map.values()))} shards)")
    elif list(out_dir.glob("*.safetensors")):
        print("[ok] single-shard checkpoint (no index needed)")
    else:
        problems.append("no safetensors weights in the export")

    if not replaced:
        problems.append("ZERO tensors were replaced — the export is the unmodified base "
                        "model, i.e. the adapter had no effect")
    else:
        print(f"[ok] {len(replaced)} language tensors carry the merged LoRA")

    try:
        cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(cfg, dict) or not cfg.get("architectures"):
            problems.append("config.json is not a loadable model config")
        else:
            print("[ok] config.json loads")
    except Exception as e:  # noqa: BLE001
        problems.append(f"config.json could not be loaded: {type(e).__name__}: {e}")

    if problems:
        print("\n".join(f"[FAIL] {p}" for p in problems))
        raise ExportError("export is NOT deployable")
    print("[ok] export passes structural validation")
    print("[note] structural validity is NOT inference parity — that is decided on a GPU by "
          "training stage 9 (scripts/stage_llm.sh), then scripts/run_inference.sh.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B",
                    help="full Qwen3.5 checkpoint (hub id or local dir)")
    ap.add_argument("--adapter", required=True, help="ACE PEFT adapter dir")
    ap.add_argument("--out", required=True, help="output dense checkpoint dir")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing --out, but only after the new export has "
                         "fully validated")
    args = ap.parse_args()

    adapter = Path(args.adapter).expanduser()
    out_dir = Path(args.out).expanduser()
    # Sibling temp dir, never a suffix swap: `dense.v2` must not become `dense.tmp`.
    tmp_dir = out_dir.parent / (out_dir.name + ".tmp")

    if not adapter.is_dir():
        raise SystemExit(f"[adapter] not found: {adapter}")
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"[out] {out_dir} already exists. Pass --overwrite to replace it — the "
            f"existing export is kept until the new one has fully validated.")

    base_dir = resolve_local(args.base)
    archs = read_base_architectures(base_dir)
    if QWEN35_VLLM_ARCHITECTURE not in archs:
        raise SystemExit(
            f"[base] {base_dir} declares architectures={archs}. This tool transplants "
            f"merged weights INTO a full {QWEN35_VLLM_ARCHITECTURE} checkpoint, so the "
            f"--base must be the full Qwen3.5 release, not a text-only export.")
    print(f"[base] {base_dir} (architectures={archs})")

    if tmp_dir.exists():
        print(f"[tmp] removing stale {tmp_dir}")
        shutil.rmtree(tmp_dir)

    try:
        merged = load_merged_text_model(base_dir, adapter)
        merged_sd = merged.state_dict()
        print(f"[merge] merged text model: {len(merged_sd)} tensors")

        # Copy into the temp dir, so a failed export leaves nothing behind that could be
        # mistaken for a finished one.
        print(f"[copy] {base_dir} -> {tmp_dir}")
        shutil.copytree(base_dir, tmp_dir, symlinks=False,
                        ignore=shutil.ignore_patterns(".git*", "*.msgpack", "*.h5",
                                                      "*.bin", "*.pth", "original"))
        if not list(tmp_dir.glob("*.safetensors")):
            raise ExportError(f"no safetensors landed in {tmp_dir} — the base may ship "
                              f"weights in an unsupported format")

        full_keys = read_full_keys(tmp_dir)
        print(f"[map] full checkpoint: {len(full_keys)} tensors")

        roots = discover_language_root(list(merged_sd), full_keys)
        print(f"[map] discovered language root: {roots}")
        mapping = build_explicit_mapping(list(merged_sd), full_keys, roots)
        print(f"[map] mapped {len(mapping)}/{len(merged_sd)} text tensors")
        for t, f in sorted(mapping.items())[:2]:
            print(f"[map]   {t}  ->  {f}")

        tied = discover_tied_aliases(merged)
        if tied:
            print(f"[map] tied/shared parameters declared by the model: {sorted(tied)}")
        coverage = analyse_coverage(list(merged_sd), full_keys, mapping,
                                    tied_aliases=tied, roots=roots)
        if coverage["unmapped_unexpected"]:
            # Fail before touching a shard: a partial rewrite is worse than none.
            raise ExportError(
                f"{len(coverage['unmapped_unexpected'])} language tensor(s) have no "
                f"counterpart in the full checkpoint and are not tied aliases, so the "
                f"export would silently keep BASE weights:\n"
                + format_keys(coverage["unmapped_unexpected"]))

        replaced, mismatches = rewrite_shards(tmp_dir, merged_sd, mapping)
        del merged_sd, merged

        validate_export(tmp_dir, coverage, replaced, mismatches)

        untouched = len(full_keys) - len(replaced)
        manifest = build_manifest(
            base_model=str(base_dir), adapter=str(adapter), language_root=roots,
            source_text_tensor_count=coverage["n_source"],
            mapped_tensor_count=coverage["n_mapped"],
            tied_alias_count=len(coverage["unmapped_tied"]),
            replaced_tensor_count=len(replaced),
            full_checkpoint_tensor_count=len(full_keys),
            untouched_tensor_count=untouched,
            architecture=QWEN35_VLLM_ARCHITECTURE)
        validate_manifest(manifest, source="freshly built manifest")
        write_manifest(tmp_dir, manifest)
        print(f"\n[manifest] {MANIFEST_NAME}: {coverage['n_mapped']} mapped, "
              f"{len(coverage['unmapped_tied'])} tied, {untouched} untouched "
              f"(vision + MTP + anything else)")
    except BaseException as e:                      # includes KeyboardInterrupt
        if tmp_dir.exists():
            print(f"[cleanup] removing incomplete export {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if isinstance(e, ExportError):
            raise SystemExit(f"[export] FAILED — {e}\n"
                             f"[export] nothing was written to {out_dir}")
        raise

    # ---- publish: swap in only now that everything above succeeded --------------
    backup = None
    if out_dir.exists():
        backup = out_dir.parent / f"{out_dir.name}.old-{os.getpid()}"
        print(f"[publish] moving existing {out_dir} aside -> {backup.name}")
        out_dir.rename(backup)
    try:
        tmp_dir.rename(out_dir)
    except OSError:
        if backup is not None:
            backup.rename(out_dir)       # restore the known-good export
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)

    print(f"\n[done] dense checkpoint: {out_dir}")
    print("       Next: stage it (ADAPTER_MODE=dense; staging now REQUIRES the manifest)"
          " via training stage 9 (scripts/stage_llm.sh), then scripts/run_inference.sh.\n"
          "       Numerical parity against score_hf is what decides whether this is "
          "deployable — a structurally valid export is not yet a validated one.")


if __name__ == "__main__":
    main()