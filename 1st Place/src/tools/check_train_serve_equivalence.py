#!/usr/bin/env python
"""Show that the training-side feature code and the shipped inference copy compute the same thing.

The shipped ``assets/magnificat/`` modules are a vendored copy of the feature code, several of them
with docstrings and comments stripped, so a plain ``diff`` against ``src/training/src/magnificat/``
is mostly noise. This compares the two after parsing: ASTs with docstrings blanked. A module is
reported ``EQUIVALENT`` when the ASTs are identical. Every module whose logic differs is listed in
``EXPECTED_DIFFS`` below with the sha256 of its normalised diff, and every module that exists in
only one copy is listed in ``INFERENCE_ONLY`` or ``TRAINING_ONLY``, each with a one-line reason.
Anything else is reported as UNEXPECTED, and so is an expected difference that has disappeared
(the lists would then be out of date). The script exits 1 on any UNEXPECTED line and 0 otherwise.

    python src/tools/check_train_serve_equivalence.py      # from the repository root

The diff hashes were taken under Python 3.12, whose ``ast.unparse`` writes the normalised source.
Another Python version may print the same code differently and so report UNEXPECTED hashes; use
the Python 3.12 environment of README Setup step 2.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVE = REPO / "models" / "submission" / "assets" / "magnificat"
TRAIN = REPO / "src" / "training" / "src" / "magnificat"

#: module -> (sha256 of its normalised diff, one-line reason)
EXPECTED_DIFFS = {
    "data.py": (
        "d1a79b52e66da609d1c674a0ded417ebd5ac7f327d3ee5e362072e7923b79c8e",
        "training copy only: default_store() falls back to an unzipped data/train_transcripts/ "
        "folder; main.py never calls default_store()"),
    "dialogue_features.py": (
        "06ca9e198fddff06f3f294c545eaff27ed980d0ae768240566eb45ab61d2adb0",
        "`len(w) <= 3 and w` vs `0 < len(w) <= 3`: the same predicate"),
    "features.py": (
        "adb3a6897caa6b651da655a8f27aa15b25fcb44bd37097b2aa3b9e3b4a81658c",
        "isinstance(x, str) guards for missing objective text and .astype(int) before summing "
        "booleans"),
    "nto_features.py": (
        "1a56102763d8194b2a9cc3a9e1e0149a624cb6421f5c7aa7583c8e0b64b16bb1",
        "different context window for the NTO classifier; the ntodst_ columns are dropped by "
        "the booster"),
    "provenance.py": (
        "a92cca330562c583674f4e21fa4d7d20fe43bc8270f4e5161768676313ef1e1b",
        "three internal corpus names added to SHIPPABLE; not called by the shipped ensemble run"),
    "sandpiper_live_infer.py": (
        "dff8634640f874ecb6122d1f51e0576efc046c6c09bf0de683655422bc8486df",
        "an SO_OFF=1 switch; main.py imports the module only for the unused model_g_opt kind"),
}
#: modules shipped in assets/magnificat/ with no training twin
INFERENCE_ONLY = {"llmkt_infer.py", "model_a.py", "tsl_features.py"}
#: modules of the training package that the shipped copy does not include
TRAINING_ONLY = {"taxonomy_v2_prompts.py", "vocab_order.py", "xgb_variants.py"}


def normalised(path: Path) -> list[str]:
    """Return the source of a module as ``ast.unparse`` writes it, with every docstring blanked.

    Comments and formatting disappear in the round trip, so two copies that differ only in those
    give the same lines.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body[0].value.value = ""
    return ast.unparse(tree).splitlines()


def logic_diff(serve: Path, train: Path) -> list[str]:
    """Diff the normalised sources of the two copies of one module.

    Args:
        serve: the shipped copy (``models/submission/assets/magnificat/<name>``).
        train: the training copy (``src/training/src/magnificat/<name>``).

    Returns:
        The changed lines of a zero-context unified diff, ``-`` for the shipped copy and ``+``
        for the training copy; empty when the two are equivalent.
    """
    a, b = normalised(serve), normalised(train)
    return [ln for ln in difflib.unified_diff(a, b, "assets", "training", n=0, lineterm="")
            if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]


def diff_sha(diff: list[str]) -> str:
    """sha256 (hex) of the diff lines joined by newlines, the value stored in ``EXPECTED_DIFFS``."""
    return hashlib.sha256("\n".join(diff).encode()).hexdigest()


def main() -> int:
    """Compare every module of the two copies and print one line per module and a summary.

    Returns:
        1 if any line is UNEXPECTED, otherwise 0 (used as the exit status). Writes no files.
    """
    if sys.version_info[:2] != (3, 12):
        print(f"WARN: Python {sys.version.split()[0]}; the expected diff hashes were taken under "
              "3.12, so a hash mismatch below may only be ast.unparse formatting")
    serve_names = {p.name for p in SERVE.glob("*.py")}
    train_names = {p.name for p in TRAIN.glob("*.py")}
    unexpected = 0
    for name in sorted(serve_names | train_names):
        if name not in train_names:
            ok = name in INFERENCE_ONLY
            print(f"{name:28s} "
                  + ("inference-only (no training twin)" if ok else "UNEXPECTED: only in the shipped copy"))
            unexpected += not ok
            continue
        if name not in serve_names:
            ok = name in TRAINING_ONLY
            print(f"{name:28s} "
                  + ("training-only (not shipped)" if ok else "UNEXPECTED: only in the training copy"))
            unexpected += not ok
            continue
        diff = logic_diff(SERVE / name, TRAIN / name)
        want = EXPECTED_DIFFS.get(name)
        if not diff:
            if want:
                print(f"{name:28s} UNEXPECTED: EQUIVALENT, but a difference is expected "
                      "(EXPECTED_DIFFS is out of date)")
                unexpected += 1
            else:
                print(f"{name:28s} EQUIVALENT")
            continue
        got = diff_sha(diff)
        if want and got == want[0]:
            print(f"{name:28s} EXPECTED DIFF ({len(diff)} changed logic lines, sha256 {got[:12]}): "
                  f"{want[1]}")
            continue
        unexpected += 1
        print(f"{name:28s} UNEXPECTED DIFF ({len(diff)} changed logic lines, sha256 {got[:12]}"
              + (f", expected {want[0][:12]}" if want else ", none expected") + "):")
        for ln in diff[:12]:
            print(f"    {ln[:150]}")
    for name in sorted((INFERENCE_ONLY - serve_names) | (TRAINING_ONLY - train_names)
                       | (set(EXPECTED_DIFFS) - (serve_names & train_names))):
        print(f"{name:28s} UNEXPECTED: listed in this script but missing")
        unexpected += 1
    print(f"{'PASS' if not unexpected else 'FAIL'}: {len(serve_names & train_names)} shared modules, "
          f"{len(EXPECTED_DIFFS)} with an expected difference, {unexpected} unexpected")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
