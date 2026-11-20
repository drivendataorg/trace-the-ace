"""models/weights_manifest.json describes the scored archive, and the repo matches it."""
import os
import zlib

import pytest

from conftest import ROOT


def crc32(path, chunk=1 << 24):
    c = 0
    with open(path, "rb") as f:
        while b := f.read(chunk):
            c = zlib.crc32(b, c)
    return f"{c & 0xFFFFFFFF:08x}"


def test_manifest_shape(manifest):
    assert len(manifest["files"]) == 15
    assert len(manifest["retained"]) == 104
    paths = [e["path"] for e in manifest["files"] + manifest["retained"]]
    assert len(set(paths)) == 119
    assert all(p.startswith("inference/") and ".." not in p for p in paths)
    assert all(p.endswith(".safetensors") for p in (e["path"] for e in manifest["files"]))


def test_weights_come_from_one_archive(manifest):
    wa = manifest["weights_archive"]
    assert wa["url"].startswith("https://") and wa["url"].endswith(".zip")
    assert isinstance(wa["size_bytes"], int) and wa["size_bytes"] > sum(
        e["size_bytes"] for e in manifest["files"])
    assert len(wa["sha256"]) == 64 and int(wa["sha256"], 16) >= 0
    for e in manifest["files"]:
        assert e["source"] == {"type": "weights_archive",
                               "filename": e["path"].removeprefix("inference/")}


def test_base_shard_origins_are_pinned(manifest):
    origins = [e for e in manifest["files"] if "origin" in e]
    assert len(origins) == 8                                  # Qwen3-4B x3 + Qwen3-8B x5
    for e in origins:
        assert e["origin"]["repo"] in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
        assert len(e["origin"]["revision"]) == 40, e["path"]  # full commit hash, not a branch


def test_only_the_dense_manifest_differs_from_the_archive(manifest):
    changed = [e for e in manifest["files"] + manifest["retained"] if "original_crc32" in e]
    assert [e["path"] for e in changed] == [
        "inference/assets/llm_q35/base/dense_export_manifest.json"]
    assert changed[0]["original_crc32"] == "026b129c"


@pytest.mark.parametrize("i", range(104))
def test_retained_file_matches(manifest, i):
    e = manifest["retained"][i]
    p = ROOT / e["path"]
    assert p.is_file(), f"missing {e['path']}"
    assert p.stat().st_size == e["size_bytes"], e["path"]
    assert crc32(p) == e["crc32"], e["path"]


def test_no_unlisted_files_in_inference(manifest):
    listed = {e["path"] for e in manifest["files"] + manifest["retained"]}
    skip = {"inference/data", "inference/submission.csv"}   # run_inference.sh scratch
    found = {str(p.relative_to(ROOT)).replace(os.sep, "/")
             for p in (ROOT / "inference").rglob("*")
             if (p.is_file() or p.is_symlink()) and "__pycache__" not in p.parts}
    assert sorted(found - listed - skip) == []


@pytest.mark.skipif(os.environ.get("ACE_VERIFY_WEIGHTS") != "1",
                    reason="set ACE_VERIFY_WEIGHTS=1 to CRC-check the 44.5 GB of weights")
def test_weight_files_match(manifest):
    bad = []
    for e in manifest["files"]:
        p = ROOT / e["path"]
        if not p.exists() or p.stat().st_size != e["size_bytes"] or crc32(p) != e["crc32"]:
            bad.append(e["path"])
    assert bad == []
