#!/usr/bin/env python
"""Assemble a submission tree from training outputs, under the names main.py expects.

Trained artifacts come from ``build/workspace/`` (the pipeline's outputs, or the shipped intermediate files);
everything that is code or configuration rather than a training output (main.py, the vendored
``magnificat`` package, the tokenizer, ``model_kind.txt``, ``blend_weight.txt``, ``v2_spec.txt``)
is copied from this repository's shipped tree. Each staged file is then compared against the
shipped ``assets/`` and reported as identical or not.

    python src/tools/stage_assets.py --training-root build/workspace --out build/workspace/build/submission

Follow with ``bash src/tools/package_submission.sh <out> <zip>`` to write the manifest and the zip.

A trained file missing from the training root is taken from ``artifacts/_bundle/`` (where
``reproduce.py`` moves a shipped file before a step overwrites it) and printed as a ``FALLBACK``
line: the staged tree then holds the shipped file, not a rebuilt one, so the script exits 1 after
staging. ``--allow-fallback`` accepts that and exits 0. A file in neither place is an error.

``--out`` is deleted and re-created, so the script refuses any ``--out`` that is or contains the
shipped tree ``models/submission``, that contains the repository or the training root, or that lies
inside the repository but outside ``build/``. Outside the repository it only replaces an empty
directory or one it staged before (marked by a ``.staged_by_stage_assets`` file).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "models" / "submission"            # the shipped submission tree (the scored one with re-saved vocabularies and edited comments; models/README.md)
SM = "external/dialogue-kt/saved_models/lmkt_mrft_sa_8b_all35k"

#: training output (relative to the training root, build/workspace/)  ->  name inside assets/
TRAINED = {
    "artifacts/baseline_v2shrunkD.joblib": "baseline.joblib",
    "artifacts/turn_correctness_clf.joblib": "turn_correctness_clf.joblib",
    "artifacts/nto_moves_clf.joblib": "nto_moves_clf.joblib",
    "artifacts/stu_states_clf.joblib": "stu_states_clf.joblib",
    "artifacts/doc29_clf.joblib": "doc29_clf.joblib",
    "artifacts/v2t_clf_v2ext_122bC.joblib": "v2t_clf.joblib",
    "artifacts/v2s_clf_v2stu_122bC.joblib": "v2s_clf.joblib",
    "artifacts/llmkt_calibrator_mrft_sub0808.joblib": "llmkt_calibrator.joblib",
    "artifacts/nc_curriculum.csv": "nc_curriculum.csv",
    f"{SM}/adapter_model.safetensors": "lora_llm/adapter_model.safetensors",
    f"{SM}/adapter_config.json": "lora_llm/adapter_config.json",
}
#: code / configuration copied verbatim from the shipped tree
STATIC_TOP = ["main.py", "main_llmkt_lora.py"]
STATIC_ASSETS = ["model_kind.txt", "blend_weight.txt", "v2_spec.txt", "tokenizer", "magnificat"]


def sha256(p: Path) -> str:
    """sha256 (hex) of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


MARKER = ".staged_by_stage_assets"


def check_out(out: Path, tr: Path) -> None:
    """Refuse an ``--out`` whose deletion would remove shipped or unrelated files.

    Args:
        out: the resolved output directory, which ``main`` deletes and re-creates.
        tr: the resolved training root.

    Returns:
        Nothing; exits with an explanation if ``out`` is or contains the shipped tree, the
        repository, the training root or the default workspace, lies inside the repository
        but outside ``build/``, or is a non-empty directory elsewhere without the
        ``.staged_by_stage_assets`` marker.
    """
    repo, shipped, build = REPO.resolve(), SHIPPED.resolve(), (REPO / "build").resolve()
    default_ws = (REPO / "build" / "workspace").resolve()   # the pipeline's default working directory
    problem = None
    if out == shipped or out.is_relative_to(shipped):
        problem = "it is (inside) the shipped tree models/submission"
    elif shipped.is_relative_to(out) or repo.is_relative_to(out):
        problem = "it contains the repository or its shipped tree"
    elif tr.is_relative_to(out):
        problem = f"it contains the training root {tr}"
    elif out == default_ws or default_ws.is_relative_to(out):
        problem = "it is or contains the default working directory build/workspace"
    elif out.is_relative_to(repo) and not (out.is_relative_to(build) and out != build):
        problem = "it is inside the repository but not under build/"
    elif (not out.is_relative_to(repo) and out.is_dir() and any(out.iterdir())
          and not (out / MARKER).is_file()):
        problem = f"it is a non-empty directory that this tool did not stage (no {MARKER})"
    if problem:
        raise SystemExit(f"stage_assets.py: refusing --out {out}: {problem}. --out is deleted and "
                         f"re-created; pass a new directory, e.g. build/workspace/build/submission.")


def main() -> None:
    """Stage a submission tree from the training outputs and compare it with the shipped assets.

    Returns:
        Nothing; writes ``<out>/main.py``, ``<out>/main_llmkt_lora.py``, ``<out>/assets/``
        (trained files under their shipped names, the static files, and
        ``assets/lora_sha256.txt``) and the marker file, and prints IDENTICAL or DIFFERENT per
        asset. Exits 1 if a trained file was taken from ``artifacts/_bundle/`` (FALLBACK) and
        ``--allow-fallback`` was not given, or if a trained file is missing from both places.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, default=REPO / "build" / "workspace")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-fallback", action="store_true",
                    help="exit 0 even when a trained file came from artifacts/_bundle/ (the shipped "
                         "copy) because the training root has none")
    args = ap.parse_args()
    out, tr = args.out.resolve(), args.training_root.resolve()
    check_out(out, tr)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets" / "lora_llm").mkdir(parents=True)
    (out / MARKER).write_text("staged by src/tools/stage_assets.py; safe to delete\n")

    for name in STATIC_TOP:
        shutil.copy2(SHIPPED / name, out / name)
    for name in STATIC_ASSETS:
        src = SHIPPED / "assets" / name
        if src.is_dir():
            shutil.copytree(src, out / "assets" / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, out / "assets" / name)

    fallbacks = []
    for rel, name in TRAINED.items():
        src = tr / rel
        if not src.exists():  # reproduce.py parks bundled originals here before overwriting
            src = tr / "artifacts" / "_bundle" / rel
            if not src.exists():
                raise SystemExit(f"stage_assets.py: {rel} is neither in {tr} nor in "
                                 f"{tr / 'artifacts' / '_bundle'}; re-run the step that writes it")
            fallbacks.append(f"FALLBACK  assets/{name}  <- artifacts/_bundle/{rel} (the shipped copy: "
                             f"the step that writes {rel} moved it aside and did not write it again)")
        shutil.copy2(src, out / "assets" / name)

    # main.py re-hashes the adapter against this file before loading it
    lora = out / "assets" / "lora_llm"
    (out / "assets" / "lora_sha256.txt").write_text(
        "".join(f"{sha256(lora / n)}  {n}\n" for n in ("adapter_config.json", "adapter_model.safetensors")))

    print(f"staged {out}")
    n_same = 0
    for p in sorted((out / "assets").rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(out / "assets")
        ref = SHIPPED / "assets" / rel
        same = ref.exists() and sha256(ref) == sha256(p)
        n_same += same
        print(f"  {'IDENTICAL' if same else 'DIFFERENT'}  assets/{rel}")
    print(f"{n_same} files byte-identical to the shipped assets/")
    for ln in fallbacks:
        print(ln)
    if fallbacks and not args.allow_fallback:
        raise SystemExit(f"stage_assets.py: {len(fallbacks)} trained file(s) staged from the shipped "
                         "copy, not from the training root (FALLBACK lines above); pass "
                         "--allow-fallback to accept that")


if __name__ == "__main__":
    main()
