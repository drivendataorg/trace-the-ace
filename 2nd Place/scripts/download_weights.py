#!/usr/bin/env python
"""Fetch the 15 model weight files into inference/assets/ and verify the whole tree.

All 15 files (44.5 GB) are one ZIP, `weights_archive.url` in models/weights_manifest.json. By
default it is downloaded into --cache-dir (resumable, stdlib only), checked against its size and
SHA-256, extracted, and deleted unless --keep-archive (peak disk ~90 GB, ~45 GB after). Every
file, weights and repository files alike, is checked against the scored archive's size + CRC32.
Other sources: --from-archive ZIP_OR_DIR (this ZIP, the scored submission ZIP or an extracted
folder; --link to link instead of copy) and --base-from-hf (the 8 Qwen3-4B/8B base shards from
Hugging Face at their pinned revision, the other 7 files by HTTP Range reads of the remote ZIP).
Only the 15 `files` entries are ever written. Inference afterwards needs no network.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "weights_manifest.json"
INFERENCE = ROOT / "inference"
CHUNK = 1 << 24


def crc32(path: Path, chunk: int = CHUNK) -> str:
    c = 0
    with open(path, "rb") as f:
        while b := f.read(chunk):
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def verify(path: Path, entry: dict) -> str | None:
    """None if OK, else a reason."""
    if not path.exists():
        return "missing"
    if path.stat().st_size != entry["size_bytes"]:
        return f"size {path.stat().st_size} != {entry['size_bytes']}"
    got = crc32(path)
    if got != entry["crc32"]:
        return f"crc32 {got} != {entry['crc32']}"
    return None


def member_name(entry: dict) -> str:
    """inference/assets/x -> assets/x: the member path inside either archive."""
    return Path(entry["path"]).relative_to(INFERENCE.relative_to(ROOT)).as_posix()


def _place(dest: Path) -> Path:
    """Prepare dest for writing; returns the temporary path to write to."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    return dest.with_suffix(dest.suffix + ".part")


# ------------------------------------------------------------------ archive readers
class Archive:
    """The 15 weight files inside a ZIP (local file or any seekable stream) or a folder."""

    def __init__(self, src, link: bool = False):
        self.link = link
        if isinstance(src, Path) and src.is_dir():
            if not (src / "assets").is_dir():
                raise SystemExit(f"--from-archive {src}: no assets/ folder there")
            self.dir, self.zip = src, None
        else:
            if link:
                raise SystemExit("--link needs an extracted folder, not a ZIP")
            self.dir, self.zip = None, zipfile.ZipFile(src)
            self.names = {i.filename for i in self.zip.infolist()}

    def has(self, entry: dict) -> bool:
        if self.zip is not None:
            return member_name(entry) in self.names
        return (self.dir / member_name(entry)).is_file()

    def extract(self, entry: dict, dest: Path) -> None:
        tmp = _place(dest)
        if self.zip is not None:
            with self.zip.open(member_name(entry)) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w, CHUNK)   # zipfile raises on a CRC mismatch
        elif self.link:
            src = self.dir / member_name(entry)
            try:
                os.link(src, dest)                # same filesystem: hard link
            except OSError:
                dest.symlink_to(src.resolve())    # otherwise a symlink
            return
        else:
            shutil.copyfile(self.dir / member_name(entry), tmp)
        tmp.replace(dest)

    def close(self) -> None:
        if self.zip is not None:
            self.zip.close()


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests, so zipfile can
    read the central directory and single members without fetching the whole ZIP."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                raise OSError(f"{self.url}: server ignored the Range request (HTTP {r.status})")
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


# ------------------------------------------------------------------ download
def download(url: str, dest: Path, size: int, digest: str) -> None:
    """Resumable download of url to dest (via dest.part), then size + SHA-256 check."""
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"download {url}\n  -> {dest}  ({size / 1e9:.2f} GB"
              + (f", resuming at {have / 1e9:.2f} GB)" if have else ")"), flush=True)
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:            # nothing left to fetch
                r = None
            else:
                raise SystemExit(f"!! download failed: HTTP {e.code} for {url}")
        if r is not None:
            with r:
                if have and r.status != 206:      # server ignored Range: start over
                    print("  server does not support resume; restarting", flush=True)
                    have = 0
                mode = "ab" if have else "wb"
                t0, last, got = time.time(), 0.0, have
                with open(part, mode) as f:
                    while b := r.read(1 << 20):
                        f.write(b)
                        got += len(b)
                        now = time.time()
                        if now - last >= 5 or got == size:
                            rate = (got - have) / max(now - t0, 1e-9) / 1e6
                            print(f"  {got / 1e9:6.2f} / {size / 1e9:.2f} GB  "
                                  f"({100 * got / size:5.1f} %, {rate:.0f} MB/s)", flush=True)
                            last = now
    got = part.stat().st_size
    if got != size:
        raise SystemExit(f"!! download incomplete: {got} of {size} bytes; re-run to resume")
    print("  checking SHA-256 ...", flush=True)
    h = sha256(part)
    if h != digest:
        part.unlink()
        raise SystemExit(f"!! SHA-256 mismatch for {url}: got {h}, expected {digest} "
                         f"(partial file removed)")
    part.replace(dest)


def fetch_base_from_hf(entry: dict, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    o = entry["origin"]
    cached = hf_hub_download(repo_id=o["repo"], filename=Path(entry["path"]).name,
                             revision=o["revision"])
    tmp = _place(dest)
    shutil.copyfile(cached, tmp)   # copy out of the HF cache: the tree must be self-contained
    tmp.replace(dest)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true",
                    help="check the whole inference tree (code + weights), download nothing")
    ap.add_argument("--from-archive", metavar="ZIP_OR_DIR", type=Path,
                    help="local weights ZIP, scored submission ZIP, or an extracted folder "
                         "holding assets/: take the weight files from there")
    ap.add_argument("--link", action="store_true",
                    help="with --from-archive DIR: hard-link (same filesystem) or symlink "
                         "instead of copying")
    ap.add_argument("--weights-url", default=os.environ.get("WEIGHTS_URL"),
                    help="URL of the weights ZIP (default: weights_archive.url in the "
                         "manifest; env WEIGHTS_URL)")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "weights",
                    help="where the downloaded ZIP is kept while extracting "
                         "(default: .cache/weights/)")
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the downloaded ZIP after a verified extraction "
                         "(default: delete it)")
    ap.add_argument("--base-from-hf", action="store_true",
                    help="take the 8 Qwen3-4B/8B base shards (24.4 GB) from Hugging Face "
                         "(pinned revision) and only the other 7 members from the remote ZIP")
    a = ap.parse_args()
    if a.link and not a.from_archive:
        ap.error("--link requires --from-archive")
    if a.from_archive and not a.from_archive.exists():
        ap.error(f"--from-archive {a.from_archive}: not found")
    if a.from_archive and a.base_from_hf:
        ap.error("--base-from-hf downloads; it cannot be combined with --from-archive")
    m = json.loads(MANIFEST.read_text())
    wa = m["weights_archive"]
    url = a.weights_url or wa["url"]

    todo, bad = [], []
    for e in m["files"]:
        why = verify(ROOT / e["path"], e)
        if why is None:
            print(f"ok      {e['path']}")
        elif a.verify_only:
            bad.append((e["path"], why))
        else:
            todo.append((e, why))

    downloaded, from_hf = None, []
    if todo:
        archive = None
        try:
            if a.from_archive:
                archive = Archive(a.from_archive, link=a.link)
            elif a.base_from_hf:
                from_hf = [t for t in todo if "origin" in t[0]]
                for e, why in from_hf:
                    print(f"hf      {e['path']}  ({why})", flush=True)
                    fetch_base_from_hf(e, ROOT / e["path"])
                todo = [t for t in todo if "origin" not in t[0]]
                if todo:
                    print(f"remote  {len(todo)} member(s) from {url}", flush=True)
                    archive = Archive(io.BufferedReader(HTTPRangeFile(url, wa["size_bytes"]),
                                                        buffer_size=CHUNK))
            else:
                downloaded = a.cache_dir / Path(url.split("?")[0]).name
                if not (downloaded.exists() and downloaded.stat().st_size == wa["size_bytes"]
                        and sha256(downloaded) == wa["sha256"]):
                    download(url, downloaded, wa["size_bytes"], wa["sha256"])
                archive = Archive(downloaded)
            for e, why in todo:
                dest = ROOT / e["path"]
                if archive is None or not archive.has(e):
                    bad.append((e["path"], "not in the archive"))
                    continue
                print(f"extract {e['path']}  ({e['size_bytes'] / 1e9:.2f} GB, was: {why})",
                      flush=True)
                try:
                    archive.extract(e, dest)
                except zipfile.BadZipFile as err:
                    bad.append((e["path"], f"corrupt member: {err}"))
                    continue
        finally:
            if archive is not None:
                archive.close()
        for e, _ in todo + from_hf:
            if any(p == e["path"] for p, _ in bad):
                continue
            why = verify(ROOT / e["path"], e)
            if why:
                bad.append((e["path"], why))
            else:
                print(f"ok      {e['path']}")

    for e in m["retained"]:                      # the small files shipped in this repo
        why = verify(ROOT / e["path"], e)
        if why:
            bad.append((e["path"], why))

    n = len(m["files"]) + len(m["retained"])
    if bad:
        print(f"\n{len(bad)} of {n} files FAILED verification:")
        for p, why in bad:
            print(f"  {p}: {why}")
        if downloaded is not None and downloaded.exists():
            print(f"(downloaded archive kept for inspection: {downloaded})")
        return 1
    if downloaded is not None and downloaded.exists() and not a.keep_archive:
        downloaded.unlink()
        print(f"removed {downloaded} (verified extraction; --keep-archive keeps it)")
    print(f"\nall {n} files of the scored archive verified (size + CRC32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
