#!/usr/bin/env python
"""Pack the 15 weight files of the scored submission into one ZIP for hosting (author side).

Input: a verified inference/assets/ tree and models/weights_manifest.json; only the `files`
entries are packed, never the `retained` ones. Layout as in the scored archive (so
--from-archive accepts either ZIP): WEIGHTS_MANIFEST.json + assets/<path inside inference/assets>.
Members are stored (safetensors do not compress), ZIP64, with a fixed timestamp, so the same
inputs give the same bytes. Inputs are checked against the manifest before writing, and every
member is re-read before the size and SHA-256 are printed.

    python scripts/make_weights_archive.py [--out .cache/ace_models_weights.zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "weights_manifest.json"
INFERENCE = ROOT / "inference"
ARCHIVE_MANIFEST = "WEIGHTS_MANIFEST.json"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)
CHUNK = 1 << 24


def crc32(path: Path) -> str:
    c = 0
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def member_name(entry: dict) -> str:
    """inference/assets/x -> assets/x (a path inside the scored archive)."""
    return Path(entry["path"]).relative_to(INFERENCE.relative_to(ROOT)).as_posix()


def zinfo(name: str, size: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    zi.compress_type = zipfile.ZIP_STORED
    zi.external_attr = 0o644 << 16
    zi.file_size = size
    return zi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / ".cache" / "ace_models_weights.zip",
                    help="output ZIP (default: .cache/ace_models_weights.zip, git-ignored)")
    a = ap.parse_args()
    files = json.loads(MANIFEST.read_text())["files"]
    total = sum(e["size_bytes"] for e in files)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(a.out.parent).free
    print(f"{len(files)} files, {total / 1e9:.2f} GB; free at {a.out.parent}: {free / 1e9:.1f} GB")
    if free < total * 1.02:
        print("!! not enough free disk space for the archive")
        return 2

    print("== checking inputs against the manifest (size + CRC32)")
    for e in files:
        src = (ROOT / e["path"]).resolve()     # read symlinked weights from their target
        why = ("missing" if not src.is_file()
               else f"size {src.stat().st_size} != {e['size_bytes']}"
               if src.stat().st_size != e["size_bytes"]
               else None)
        if why is None and (got := crc32(src)) != e["crc32"]:
            why = f"crc32 {got} != {e['crc32']}"
        if why:
            print(f"!! {e['path']}: {why} (run scripts/download_weights.py first)")
            return 1
        print(f"ok      {e['path']}", flush=True)

    listing = {"files": [{"path": member_name(e), "size_bytes": e["size_bytes"],
                          "crc32": e["crc32"]} for e in files]}
    tmp = a.out.with_suffix(a.out.suffix + ".part")
    print(f"== writing {a.out}")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        body = (json.dumps(listing, indent=2) + "\n").encode()
        z.writestr(zinfo(ARCHIVE_MANIFEST, len(body)), body)
        for e in sorted(files, key=member_name):
            with open((ROOT / e["path"]).resolve(), "rb") as f, \
                    z.open(zinfo(member_name(e), e["size_bytes"]), "w", force_zip64=True) as w:
                shutil.copyfileobj(f, w, CHUNK)
            print(f"packed  {member_name(e)}", flush=True)

    print("== re-reading every member")
    with zipfile.ZipFile(tmp) as z:
        infos = {i.filename: i for i in z.infolist()}
        expected = {member_name(e): e for e in files}
        if set(infos) != set(expected) | {ARCHIVE_MANIFEST}:
            print(f"!! unexpected member set: {sorted(set(infos) ^ (set(expected) | {ARCHIVE_MANIFEST}))}")
            return 1
        for name, e in expected.items():
            i = infos[name]
            c = 0
            with z.open(i) as r:                  # zipfile also checks the stored CRC
                while b := r.read(CHUNK):
                    c = zlib.crc32(b, c)
            if i.file_size != e["size_bytes"] or f"{c & 0xFFFFFFFF:08x}" != e["crc32"]:
                print(f"!! {name}: re-read does not match the manifest")
                return 1
    tmp.replace(a.out)
    size, digest = a.out.stat().st_size, sha256(a.out)
    print(f"\n{a.out}\nsize_bytes {size}\nsha256     {digest}")
    print(json.dumps({"size_bytes": size, "sha256": digest}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
