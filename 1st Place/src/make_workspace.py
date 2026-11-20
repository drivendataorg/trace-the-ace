#!/usr/bin/env python
"""Assemble the training pipeline's working directory, ``build/workspace/``.

The pipeline (``src/training/reproduce.py`` and the scripts it calls) runs inside one directory
laid out the way it was when the model was trained: ``artifacts/``, ``data/``,
``external/dialogue-kt/{data,results,saved_models}/``. This repository instead stores
those files by meaning (``data/interim/``, ``data/external/``, ``models/``), and
it contains **no competition data**. This script rebuilds the working directory from the two:

0. checks that ``data/raw/`` (or ``--raw``) holds ``train_features*.csv``, ``train_labels*.csv`` and
   the transcripts (``train_transcripts*.zip`` or an unzipped ``train_transcripts/`` folder) before
   it writes anything;
1. copies the training code (``src/training/``) into ``build/workspace/``;
2. copies every shipped intermediate file to the place the pipeline expects it, following
   ``src/training/workspace_map.tsv`` (one row per file: workspace path, repository path, sha256).
   A file that is already there is left alone; with ``--force`` it is replaced, and each replaced
   file that differed from the shipped one is printed (``OVERWROTE``);
3. links the raw competition data from ``data/raw/`` (or ``--raw``) into ``build/workspace/data/`` and unpacks the
   transcripts (relative links when both folders are inside the repository, so moving the
   repository keeps them valid). ``--link-only`` does just this step; ``src/run_training.py``
   runs it on every call;
4. puts the training labels back into the two out-of-fold prediction files they were removed from
   (from ``train_labels.csv``), and rebuilds ``mrft_oof_3fold.csv`` from its keyed copy. Each restored
   text is checked against the sha256 of the original before it is written: the restored files are
   byte-identical to the ones the model was trained with. A file that is already there and differs
   (for example your own re-run of step 05a, or an out-of-fold file you pooled after re-running
   06c) is KEPT, unless ``--force`` is given (then ``OVERWROTE``).

``--force`` therefore means "start again from the shipped files": it discards the outputs of the
steps you re-ran, and also the shipped copies that ``reproduce.py`` moved to ``artifacts/_bundle/``.

A full assembly ends by writing ``build/workspace/.workspace_ready``, which records the raw-data
folder it used. Later calls without ``--raw`` (including ``--link-only`` and the calls made by
``src/run_training.py``) use that recorded folder, so a workspace assembled with
``--raw /path/to/competition/files`` keeps reading from there; without a recorded folder they use
``data/raw/``.

Two files are not shipped because they consist of competition transcripts; ``reproduce.py`` steps
01a and 01b regenerate them from ``data/raw/`` (``traceace_full.pkl``, ``turn_text.pkl.gz``).

    python src/make_workspace.py                  # -> build/workspace/
    python src/make_workspace.py --raw /path/to/competition/files --force
    python src/make_workspace.py --link-only      # re-link the raw data only (recorded folder or data/raw/)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "src" / "training"
MAP = CODE / "workspace_map.tsv"
MARK_NAME = ".workspace_ready"   # written after a full assembly; records the raw-data folder used


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def find_one(raw: Path, prefix: str, suffix: str) -> Path:
    """Return the single file in ``raw`` named ``<prefix>*<suffix>``.

    Args:
        raw: folder with the raw competition files.
        prefix: start of the file name, e.g. ``"train_labels"``.
        suffix: end of the file name, e.g. ``".csv"``.

    Returns:
        The path of the matching file. Exits with an error message unless exactly one file matches.
    """
    hits = sorted(p for p in raw.glob(f"{prefix}*{suffix}") if p.is_file())
    if len(hits) != 1:
        sys.exit(f"expected one {prefix}*{suffix} in {raw}, found {len(hits)}: {[h.name for h in hits]}")
    return hits[0]


def raw_problems(raw: Path) -> list[str]:
    """List what is missing from the raw data folder (checked before anything is written).

    Args:
        raw: folder that should hold ``train_features*.csv``, ``train_labels*.csv`` and the train
            transcripts (``train_transcripts*.zip`` or a ``train_transcripts/`` folder).

    Returns:
        One message per problem; an empty list when the folder is complete.
    """
    out = []
    for prefix, suffix in (("train_features", ".csv"), ("train_labels", ".csv")):
        hits = [p.name for p in raw.glob(f"{prefix}*{suffix}") if p.is_file()]
        if len(hits) != 1:
            out.append(f"expected one {prefix}*{suffix}, found {len(hits)}: {hits}")
    zips = [p.name for p in raw.glob("train_transcripts*.zip") if p.is_file()]
    if not (raw / "train_transcripts").is_dir() and len(zips) != 1:
        out.append(f"expected a train_transcripts/ folder or one train_transcripts*.zip, found {zips}")
    return out


def link_target(target: Path, link: Path, relative: bool) -> str:
    """Return the text of a symlink at ``link`` pointing to ``target`` (relative or absolute)."""
    return os.path.relpath(target, link.parent) if relative else str(target)


def link_raw(raw: Path, out: Path, strict: bool) -> None:
    """Step 3: (re-)link the raw competition files into ``<out>/data/`` and unpack the transcripts.

    Stale or dangling links are replaced. Links are relative when both folders are inside the
    repository. An unpacked ``train_transcripts/`` folder in ``raw`` is linked; otherwise the zip
    is extracted into ``<out>/data/train_transcripts/`` (again if an earlier unpack is incomplete).

    Args:
        raw: folder with the raw competition files.
        out: the workspace, normally ``build/workspace/``.
        strict: if True, a missing raw file stops the program; if False (``--link-only``) it is a
            warning and the existing links are kept.

    Returns:
        None. Creates links (and possibly the unpacked transcripts) under ``<out>/data/``.
    """
    problems = raw_problems(raw)
    if problems:
        msg = f"raw data in {raw}: " + "; ".join(problems)
        if strict:
            sys.exit(msg)
        print(f"WARNING   {msg} (existing links in {out / 'data'} kept)")
        return
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    try:                                     # both inside the repository: links survive moving it
        raw.relative_to(REPO); out.relative_to(REPO)
        relative = True
    except ValueError:
        relative = False
    zips = sorted(p for p in raw.glob("train_transcripts*.zip") if p.is_file())
    files = [find_one(raw, "train_features", ".csv"), find_one(raw, "train_labels", ".csv")] + zips[:1]
    want = {f.name: link_target(f, data / f.name, relative) for f in files}
    for prefix, suffix in (("train_features", ".csv"), ("train_labels", ".csv"), ("train_transcripts", ".zip")):
        for old in data.glob(f"{prefix}*{suffix}"):  # stale links: other file name, --raw or location
            if old.is_symlink() and os.readlink(old) != want.get(old.name):
                old.unlink()
        for old in data.iterdir():                   # dangling links (glob does not list them)
            if old.is_symlink() and not old.exists() and old.name.startswith(prefix) \
                    and old.name.endswith(suffix):
                old.unlink()
    for name, target in want.items():
        link = data / name
        if link.is_symlink():
            continue                                 # already points at the right file
        if link.exists():
            sys.exit(f"{link} is a regular file, not a link to {raw}: remove it")
        link.symlink_to(target)
    tdir = data / "train_transcripts"
    if tdir.is_symlink() and not tdir.exists():
        tdir.unlink()                        # dangling (repository or --raw moved)
    if (raw / "train_transcripts").is_dir():
        want = link_target(raw / "train_transcripts", tdir, relative)
        if tdir.is_symlink() and os.readlink(tdir) != want:
            tdir.unlink()
        if not tdir.exists():
            tdir.symlink_to(want, target_is_directory=True)
    elif not tdir.is_symlink():              # unpack the zip, again if an earlier unpack is incomplete
        with zipfile.ZipFile(zips[0]) as zf:     # the zip has no top-level folder
            n_zip = sum(1 for n in zf.namelist() if n.endswith(".csv"))
            if not tdir.is_dir() or sum(1 for _ in tdir.glob("*.csv")) < n_zip:
                tdir.mkdir(exist_ok=True)
                zf.extractall(tdir)
    n_tx = sum(1 for _ in tdir.glob("*.csv"))
    print(f"raw data  {raw} -> {data} ({n_tx} transcripts{', relative links' if relative else ''})")
    if n_tx != 22821:
        print(f"WARNING   expected 22,821 training transcripts, found {n_tx}")


def load_labels(raw: Path) -> dict[str, float]:
    """Read ``train_labels*.csv`` from ``raw`` as a ``{response_id: is_correct}`` dict (0.0/1.0)."""
    f = find_one(raw, "train_labels", ".csv")
    with f.open() as fh:
        return {r["response_id"]: float(r["is_correct"]) for r in csv.DictReader(fh)}


def strip_column(text: str, idx: int) -> str:
    """Remove field ``idx`` from every line of a quote-free CSV.

    Used once, at release, to take the label column out of the out-of-fold files; the inverse of
    ``restore_column``.

    Args:
        text: the CSV file's text (no quoted fields).
        idx: 0-based index of the field to remove.

    Returns:
        The CSV text without that field.
    """
    out = []
    for line in text.split("\n"):
        if line:
            f = line.split(",")
            del f[idx]
            line = ",".join(f)
        out.append(line)
    return "\n".join(out)


def restore_column(text: str, idx: int, fmt: str, labels: dict[str, float], key: int = 0) -> str:
    """Insert the training label as field ``idx`` (header ``y``) of a quote-free CSV.

    Args:
        text: the CSV text without its label column.
        idx: 0-based position of the inserted field.
        fmt: ``"float"`` writes ``0.0``/``1.0``; anything else writes ``0``/``1``.
        labels: ``{response_id: is_correct}`` from ``load_labels``.
        key: index of the field holding the response id.

    Returns:
        The CSV text with the label column put back.
    """
    lines = text.split("\n")
    out = []
    for n, line in enumerate(lines):
        if line:
            f = line.split(",")
            if n == 0:
                f.insert(idx, "y")
            else:
                v = labels[f[key]]
                f.insert(idx, (f"{v:.1f}" if fmt == "float" else str(int(v))))
            line = ",".join(f)
        out.append(line)
    return "\n".join(out)


def feature_order(raw: Path) -> dict[str, int]:
    """Return ``{response_id: row position}`` for ``train_features*.csv`` in ``raw``."""
    with find_one(raw, "train_features", ".csv").open() as fh:
        return {r["response_id"]: i for i, r in enumerate(csv.DictReader(fh))}


def derive_pooled(keyed_text: str, order: dict[str, int]) -> str:
    """Rebuild ``mrft_oof_3fold.csv`` from its keyed copy.

    The pooled file is the keyed file (``response_id,fold,y,p``) without its id column, with rows
    ordered by fold and then by the response's position in ``train_features.csv``: the order in
    which the three held-out folds were scored and concatenated.

    Args:
        keyed_text: text of the restored keyed file (labels already put back).
        order: ``{response_id: row position}`` from ``feature_order``.

    Returns:
        The text of ``mrft_oof_3fold.csv`` (``fold,y,p``), ending with a newline.
    """
    lines = [l for l in keyed_text.split("\n") if l]
    head, body = lines[0], lines[1:]
    body.sort(key=lambda l: (int(l.split(",")[1]), order[l.split(",")[0]]))
    return "\n".join(",".join(l.split(",")[1:]) for l in [head] + body) + "\n"


def recorded_raw(out: Path) -> Path | None:
    """Return the raw-data folder recorded in ``<out>/.workspace_ready`` by an earlier full assembly.

    Args:
        out: the workspace folder.

    Returns:
        The recorded folder (resolved against the repository root if relative), or None if there
        is no marker file or it records no folder.
    """
    try:
        for line in (out / MARK_NAME).read_text().splitlines():
            if line.startswith("raw="):
                return REPO / line[4:]       # relative to the repository unless absolute
    except OSError:
        pass
    return None


def main() -> None:
    """Assemble the workspace (steps 0-4 of the module docstring), or only re-link with ``--link-only``.

    Returns:
        None. Writes the workspace (default ``build/workspace/``) and, after a full assembly,
        ``<out>/.workspace_ready``. Exits with an error message if raw files are missing, a shipped
        file is missing from the repository, or a restored file does not match its sha256.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=None,
                    help="folder with train_features*.csv, train_labels*.csv and train_transcripts*.zip "
                         "(or an unzipped train_transcripts/ folder); default: the folder recorded in "
                         "<out>/.workspace_ready by the last full assembly, else data/raw/")
    ap.add_argument("--out", type=Path, default=REPO / "build" / "workspace")
    ap.add_argument("--force", action="store_true",
                    help="replace files that already exist in the workspace with the shipped ones "
                         "(discards the outputs of steps you re-ran; each one that differed is printed)")
    ap.add_argument("--link-only", action="store_true",
                    help="only (re-)link the raw data (step 3); a missing raw file is a warning")
    args = ap.parse_args()
    out = args.out.resolve()
    raw = (args.raw or recorded_raw(out) or REPO / "data" / "raw").resolve()
    if args.link_only:
        link_raw(raw, out, strict=False)
        return
    problems = raw_problems(raw)         # 0. before anything is written
    if problems:
        sys.exit(f"raw competition data not found in {raw}:\n  " + "\n  ".join(problems)
                 + "\nPut the files from the competition's data page there (README, Setup).")
    rows = list(csv.DictReader(MAP.open(), delimiter="\t"))

    # 1. code
    shutil.copytree(CODE, out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"code      src/training/ -> {out}")

    # 2. shipped intermediate files
    n_copy = n_kept = 0
    for r in rows:
        if r["how"] != "copy":
            continue
        src, dst = REPO / r["repo_path"], out / r["workspace_path"]
        if not src.exists():
            sys.exit(f"missing in the repository: {r['repo_path']}")
        if dst.exists():
            if not args.force:
                n_kept += 1
                if dst.stat().st_size != int(r["bytes"]):   # cheap check; --verify hashes everything
                    print(f"KEPT      {r['workspace_path']}  (differs from the shipped file: a rebuilt "
                          "output? --force replaces it)")
                continue
            if dst.stat().st_size != int(r["bytes"]) or sha256(dst) != r["sha256"]:
                print(f"OVERWROTE {r['workspace_path']}  (it differed from the shipped file)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        n_copy += 1
    print(f"copied    {n_copy} intermediate files (models, annotations, features)"
          + (f"; {n_kept} already present, left as they are" if n_kept else ""))
    bundle = out / "artifacts" / "_bundle"
    if args.force and bundle.is_dir():
        n_b = sum(1 for p in bundle.rglob("*") if p.is_file())
        shutil.rmtree(bundle)
        print(f"removed   artifacts/_bundle/ ({n_b} files: shipped copies moved aside by earlier step runs)")

    # external data the GPU steps read: MRBench (CC BY-SA 4.0), for step 02f
    mrb = REPO / "data" / "external" / "mrbench" / "raw"
    if mrb.is_dir():
        shutil.copytree(mrb, out / "external" / "datasets" / "mrbench", dirs_exist_ok=True)

    # 3. raw competition data
    link_raw(raw, out, strict=True)

    # 4. training labels back into the out-of-fold files, checked against the original sha256
    #    before anything is written; the derived file is built from the restored text, not from
    #    whatever is in the workspace
    labels = load_labels(raw)
    bad = 0
    restored: dict[str, str] = {}
    order = sorted(rows, key=lambda r: r["how"].startswith("derive:"))   # derived files last
    for r in order:
        how = r["how"]
        dst = out / r["workspace_path"]
        if how.startswith("strip-y"):
            _, fmt, idx = how.split(":")
            text = restore_column((REPO / r["repo_path"]).read_text(), int(idx), fmt, labels)
        elif how.startswith("derive:"):
            text = derive_pooled(restored[how.split(":", 1)[1]], feature_order(raw))
        else:
            continue
        restored[r["workspace_path"]] = text
        blob = text.encode()
        if hashlib.sha256(blob).hexdigest() != r["sha256"]:
            bad += 1
            print(f"MISMATCH  {r['workspace_path']}  (not written: the restored labels do not "
                  "reproduce the original)")
            continue
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(blob)
            print(f"restored  {r['workspace_path']}  (labels from train_labels)")
        elif sha256(dst) == r["sha256"]:
            print(f"ok        {r['workspace_path']}  (already restored)")
        elif args.force:
            dst.write_bytes(blob)
            print(f"OVERWROTE {r['workspace_path']}  (it differed from the original; restored)")
        else:
            print(f"KEPT      {r['workspace_path']}  (differs from the shipped file: a rebuilt output? "
                  "--force replaces it)")
    if bad:
        sys.exit(f"{bad} restored file(s) differ from the originals: is {raw} the competition's "
                 "train_labels.csv?")

    regen = [r["workspace_path"] for r in rows if r["how"].startswith("regenerate")]
    missing = [p for p in regen if not (out / p).exists()]
    print("not shipped (competition transcripts), regenerated by reproduce.py steps 01a/01b: "
          + ", ".join(Path(p).name for p in regen)
          + ("" if missing else "  [already present]"))
    try:                                 # inside the repository: recorded relative, so moving it is fine
        rec = raw.relative_to(REPO)
    except ValueError:
        rec = raw
    (out / MARK_NAME).write_text(f"assembled by src/make_workspace.py\nraw={rec}\n")
    print(f"workspace ready: {out}")


if __name__ == "__main__":
    main()
