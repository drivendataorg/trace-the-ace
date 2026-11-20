"""Weights ZIP: build (make_weights_archive.py) and fetch (download_weights.py) on a
miniature repository, served by a local HTTP server. CPU-only, no external network."""
import hashlib
import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
import zlib
from pathlib import Path

import pytest

from conftest import ROOT

WEIGHTS = {  # path inside inference/assets -> size; "base" files carry an HF origin
    "bge-base-en-v1.5/model.safetensors": 300_000,
    "llm_4b/adapters/full/adapter_model.safetensors": 70_000,
    "llm_4b/base/model-00001-of-00002.safetensors": 450_000,
    "llm_q35/base/model.safetensors-00001-of-00001.safetensors": 120_000,
}
RETAINED = {"llm_q35/base/dense_export_manifest.json": b'{"sanitised": true}\n'}


def crc(b):
    return f"{zlib.crc32(b) & 0xFFFFFFFF:08x}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(repo, *args):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("WEIGHTS_URL", None)
    return subprocess.run([sys.executable, *map(str, args)], cwd=repo, env=env,
                          capture_output=True, text=True)


# ------------------------------------------------------------------ fixtures
class RangeHandler(http.server.BaseHTTPRequestHandler):
    """Static files with HTTP Range support (http.server's own handler has none)."""
    root: Path
    ranges: list
    honour_range = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.root / self.path.lstrip("/").split("?")[0]
        if not p.is_file():
            self.send_error(404)
            return
        data = p.read_bytes()
        rng = self.headers.get("Range")
        self.ranges.append(rng)
        if rng and self.honour_range:
            a, _, b = rng.removeprefix("bytes=").partition("-")
            start, end = int(a), (int(b) if b else len(data) - 1)
            if start >= len(data):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(data)}")
                self.end_headers()
                return
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def repo(tmp_path):
    """A miniature repo: the two scripts, inference/assets with weights + one retained file."""
    r = tmp_path / "repo"
    (r / "scripts").mkdir(parents=True)
    for s in ("download_weights.py", "make_weights_archive.py"):
        shutil.copy(ROOT / "scripts" / s, r / "scripts" / s)
    rng = __import__("random").Random(0)
    files, retained = [], []
    for rel, n in WEIGHTS.items():
        b = rng.randbytes(n)
        p = r / "inference/assets" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        e = {"path": f"inference/assets/{rel}", "size_bytes": n, "crc32": crc(b),
             "source": {"type": "weights_archive", "filename": f"assets/{rel}"}}
        if "/base/model-" in rel and rel.startswith("llm_4b"):
            e["origin"] = {"repo": "Qwen/Qwen3-4B", "revision": "0" * 40}
        files.append(e)
    for rel, b in RETAINED.items():
        p = r / "inference/assets" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        retained.append({"path": f"inference/assets/{rel}", "size_bytes": len(b), "crc32": crc(b)})
    (r / "models").mkdir()
    write_manifest(r, {"description": "test", "weights_archive":
                       {"url": "http://127.0.0.1:9/unused.zip", "size_bytes": None, "sha256": None},
                       "files": files, "retained": retained})
    return r


def manifest(repo):
    return json.loads((repo / "models/weights_manifest.json").read_text())


def write_manifest(repo, m):
    (repo / "models/weights_manifest.json").write_text(json.dumps(m, indent=2))


@pytest.fixture
def built(repo, tmp_path):
    """Build the ZIP, record it in the manifest, then remove the weights from the tree."""
    out = tmp_path / "srv" / "ace_models_weights.zip"
    r = run(repo, "scripts/make_weights_archive.py", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    m = manifest(repo)
    m["weights_archive"].update(size_bytes=out.stat().st_size, sha256=sha(out))
    write_manifest(repo, m)
    for e in m["files"]:
        (repo / e["path"]).unlink()
    return out


@pytest.fixture
def server(built):
    handler = type("H", (RangeHandler,), {"root": built.parent, "ranges": []})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield handler, f"http://127.0.0.1:{srv.server_address[1]}/{built.name}"
    srv.shutdown()


def weights_ok(repo):
    return all((repo / e["path"]).is_file() and
               crc((repo / e["path"]).read_bytes()) == e["crc32"] for e in manifest(repo)["files"])


# ------------------------------------------------------------------ make_weights_archive.py
def test_archive_layout(built, repo):
    with zipfile.ZipFile(built) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        listing = json.loads(z.read("WEIGHTS_MANIFEST.json"))
    assert names[0] == "WEIGHTS_MANIFEST.json"
    assert sorted(names[1:]) == sorted(f"assets/{rel}" for rel in WEIGHTS)
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)
    assert not any("dense_export_manifest" in n for n in names)        # never retained files
    assert {f["path"]: (f["size_bytes"], f["crc32"]) for f in listing["files"]} == {
        e["source"]["filename"]: (e["size_bytes"], e["crc32"]) for e in manifest(repo)["files"]}


def test_archive_is_deterministic(repo, tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    for out in (a, b):
        assert run(repo, "scripts/make_weights_archive.py", "--out", out).returncode == 0
    assert sha(a) == sha(b)


def test_archive_refuses_a_corrupt_input(repo, tmp_path):
    p = repo / manifest(repo)["files"][0]["path"]
    p.write_bytes(b"x" + p.read_bytes()[1:])
    r = run(repo, "scripts/make_weights_archive.py", "--out", tmp_path / "bad.zip")
    assert r.returncode == 1 and "crc32" in r.stdout
    assert not (tmp_path / "bad.zip").exists()


# ------------------------------------------------------------------ download_weights.py
def test_download_extract_and_delete(repo, server):
    handler, url = server
    r = run(repo, "scripts/download_weights.py", "--weights-url", url)
    assert r.returncode == 0, r.stdout + r.stderr
    assert weights_ok(repo)
    assert not (repo / ".cache/weights/ace_models_weights.zip").exists()   # default: deleted
    assert "all 5 files" in r.stdout and "removed" in r.stdout


def test_keep_archive(repo, server, built):
    _, url = server
    r = run(repo, "scripts/download_weights.py", "--weights-url", url, "--keep-archive")
    assert r.returncode == 0, r.stdout + r.stderr
    kept = repo / ".cache/weights/ace_models_weights.zip"
    assert kept.is_file() and sha(kept) == sha(built)
    # a re-run with everything in place downloads nothing
    handler, _ = server
    n = len(handler.ranges)
    assert run(repo, "scripts/download_weights.py", "--weights-url", url).returncode == 0
    assert len(handler.ranges) == n


def test_resume_after_truncated_download(repo, server, built):
    handler, url = server
    part = repo / ".cache/weights/ace_models_weights.zip.part"
    part.parent.mkdir(parents=True)
    cut = built.stat().st_size // 3
    part.write_bytes(built.read_bytes()[:cut])
    r = run(repo, "scripts/download_weights.py", "--weights-url", url)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"bytes={cut}-" in handler.ranges and "resuming" in r.stdout
    assert weights_ok(repo)


def test_server_without_range_restarts_cleanly(repo, server, built):
    handler, url = server
    handler.honour_range = False
    part = repo / ".cache/weights/ace_models_weights.zip.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(built.read_bytes()[:1000])
    r = run(repo, "scripts/download_weights.py", "--weights-url", url)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "restarting" in r.stdout and weights_ok(repo)


def test_sha256_mismatch_fails(repo, server):
    _, url = server
    m = manifest(repo)
    m["weights_archive"]["sha256"] = "0" * 64
    write_manifest(repo, m)
    r = run(repo, "scripts/download_weights.py", "--weights-url", url)
    assert r.returncode != 0 and "SHA-256 mismatch" in (r.stdout + r.stderr)
    assert not any((repo / e["path"]).exists() for e in m["files"])
    assert not list((repo / ".cache/weights").glob("*"))      # bad bytes are not kept


def test_retained_files_are_never_overwritten(repo, built, tmp_path):
    # an archive that also carries a different copy of a retained file (like the scored ZIP)
    evil = tmp_path / "scored_like.zip"
    shutil.copy(built, evil)
    rel = next(iter(RETAINED))
    with zipfile.ZipFile(evil, "a") as z:
        z.writestr(f"assets/{rel}", b'{"original": "a developer-local path"}\n')
    before = (repo / "inference/assets" / rel).read_bytes()
    r = run(repo, "scripts/download_weights.py", "--from-archive", evil)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (repo / "inference/assets" / rel).read_bytes() == before
    assert weights_ok(repo) and evil.exists()                  # a user's ZIP is never deleted


def test_from_archive_folder_link(repo, built, tmp_path):
    folder = tmp_path / "extracted"
    with zipfile.ZipFile(built) as z:
        z.extractall(folder)
    r = run(repo, "scripts/download_weights.py", "--from-archive", folder, "--link")
    assert r.returncode == 0, r.stdout + r.stderr
    assert weights_ok(repo)
    r = run(repo, "scripts/download_weights.py", "--from-archive", built, "--link")
    assert r.returncode == 0                                    # nothing left to do
    for e in manifest(repo)["files"]:
        (repo / e["path"]).unlink()
    r = run(repo, "scripts/download_weights.py", "--from-archive", built, "--link")
    assert r.returncode != 0 and "--link needs an extracted folder" in (r.stdout + r.stderr)


def test_verify_only_reports_missing(repo, built):
    r = run(repo, "scripts/download_weights.py", "--verify-only")
    assert r.returncode == 1 and "4 of 5 files FAILED" in r.stdout


def test_remote_member_reader_reads_only_what_it_needs(repo, server, built):
    """The --base-from-hf path: members straight from the remote ZIP via HTTP Range."""
    handler, url = server
    spec = importlib.util.spec_from_file_location("dw", repo / "scripts/download_weights.py")
    dw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dw)
    size = built.stat().st_size
    remote = dw.HTTPRangeFile(url, size)
    arc = dw.Archive(io.BufferedReader(remote, buffer_size=1 << 16))
    e = next(e for e in manifest(repo)["files"] if "adapter" in e["path"])
    dest = repo / e["path"]
    arc.extract(e, dest)
    arc.close()
    assert crc(dest.read_bytes()) == e["crc32"]
    assert handler.ranges and all(r and r.startswith("bytes=") for r in handler.ranges)
    fetched = sum(int(b) - int(a) + 1 for a, b in
                  (r.removeprefix("bytes=").split("-") for r in handler.ranges))
    assert fetched < size / 2                                  # not the whole archive
