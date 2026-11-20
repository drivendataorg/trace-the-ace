#!/usr/bin/env bash
# Pack a submission tree into a zip the way the shipped zip was packed, then compare the new zip
# with models/submission.zip entry by entry.
#
#   bash src/tools/package_submission.sh <tree> <out.zip>
#   bash src/tools/package_submission.sh models/submission build/submission_check.zip   # re-pack the shipped tree
#
# Run from anywhere; the paths are resolved against the current directory.
#
# 1. For a staged tree (anything other than models/submission, e.g. step 09b's
#    build/workspace/build/submission): writes assets/manifest_sha256.txt, the sha256 of every file
#    under assets/ except the manifest, paths relative to assets/, C-locale sorted (main.py verifies
#    it on startup and aborts on any mismatch). For models/submission itself (decided by physical
#    path and file identity, so a symlinked or bind-mounted spelling counts too) the manifest is
#    left alone and every asset is checked against it instead.
# 2. Zips main.py, main_llmkt_lora.py and assets/ with `zip -r -X` (no extra attributes).
#    __pycache__ folders and .pyc files are left out of the zip; the tree is not modified (apart
#    from the regenerated manifest in a staged tree).
# 3. Compares the new zip with models/submission.zip: the two must have the same entry names
#    (files and directory entries; the expected count is read from models/submission.zip, 53),
#    and every file entry must have the same CRC-32 and size. Exits non-zero otherwise.
#    With COMPARE_REPORT_ONLY=1 (set by step 09b of the training pipeline) differing file entries
#    are listed but do not fail the run; a bad CRC or a missing/extra entry still does.
# 4. Prints the sha256 of the new zip.
#
# models/submission.zip is the scored submission with its seven text-classifier files re-saved so
# that their vocabularies are stored in sorted order, and comments and docstrings edited in 19
# source files; predictions are identical (models/README.md). Its sha256 is
# e2a8d2f28eaf3274a75413a13256feb42129ca311a538a2854078cb11dc8d2f6; the scored zip's sha256 was
# 4ab1450b1b720e1dc625e73af8c51b4dd3ad331d32937d8059c14628088dba07. A newly packed zip has a
# different sha256 from both, because zip stores file modification times, so the entry-by-entry
# comparison is the check that matters.
set -euo pipefail
# Physical paths (pwd -P) and an inode test (-ef): a symlinked home or project directory, a bind
# mount or a hard-linked copy must never route the shipped tree into the manifest-rewriting branch.
TREE=$(cd "$1" && pwd -P); OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd "$HERE/../../models/submission" && pwd -P)   # the shipped submission tree
SHIPPED_ZIP="$HERE/../../models/submission.zip"
[ -f "$SHIPPED_ZIP" ] || { echo "ERROR: $SHIPPED_ZIP not found"; exit 1; }

if [ "$TREE" -ef "$REPO" ] || [ "$TREE/assets/manifest_sha256.txt" -ef "$REPO/assets/manifest_sha256.txt" ]; then
  SHIPPED_TREE=1
else
  SHIPPED_TREE=0
fi
if [ "$SHIPPED_TREE" = 0 ]; then
  ( cd "$TREE/assets" && rm -f manifest_sha256.txt && \
    find . -name __pycache__ -type d -prune -o -type f ! -name manifest_sha256.txt ! -name '*.pyc' -printf '%P\n' | \
    LC_ALL=C sort | xargs -d '\n' sha256sum > manifest_sha256.txt )
else
  # Packing models/submission as shipped: every asset must already match the shipped manifest.
  # The usual failure is an incomplete copy: a git clone made without Git LFS leaves ~130-byte
  # pointer files where the weights should be, and main.py would then abort on the platform.
  if ! ( cd "$TREE/assets" && sha256sum -c --quiet manifest_sha256.txt ); then
    echo "ERROR: assets/ does not match assets/manifest_sha256.txt (see the FAILED lines above)."
    echo "       If the .joblib/.safetensors files are tiny, the weights are missing: re-download and"
    echo "       unpack the archive, or in a git clone run: git lfs install && git lfs pull."
    echo "       Then check the whole repository: sha256sum -c --quiet MANIFEST.sha256"
    exit 1
  fi
  echo "assets/ verified against manifest_sha256.txt"
fi
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
if command -v zip >/dev/null; then
  ( cd "$TREE" && zip -q -r -X "$OUT" main.py main_llmkt_lora.py assets -x '*__pycache__*' '*.pyc' )
else  # same entries (files + directory entries, deflate) via the standard library
  ( cd "$TREE" && python3 - "$OUT" <<'PY'
import os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for top in ("main.py", "main_llmkt_lora.py"):
        z.write(top)
    for d, dirs, files in os.walk("assets"):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        z.write(d)
        for f in sorted(files):
            if not f.endswith(".pyc"):
                z.write(os.path.join(d, f))
PY
  )
fi

python3 - "$OUT" "$SHIPPED_ZIP" <<'PY'
import sys, zipfile

new, ref = zipfile.ZipFile(sys.argv[1]), zipfile.ZipFile(sys.argv[2])
bad = new.testzip()
print("CRC OK" if bad is None else f"CRC FAIL {bad}")
a = {i.filename: i for i in new.infolist()}
b = {i.filename: i for i in ref.infolist()}
print(f"entries: {len(a)}  (models/submission.zip: {len(b)})")
missing, extra = sorted(set(b) - set(a)), sorted(set(a) - set(b))
for n in missing:
    print(f"  MISSING (in models/submission.zip, not in the new zip): {n}")
for n in extra:
    print(f"  EXTRA (in the new zip, not in models/submission.zip): {n}")
same = diff = 0
for n in sorted(set(a) & set(b)):
    if n.endswith("/"):                     # directory entries: the name is all there is
        continue
    if (a[n].CRC, a[n].file_size) == (b[n].CRC, b[n].file_size):
        same += 1
    else:
        diff += 1
        print(f"  DIFFERENT: {n}")
print(f"entry check vs models/submission.zip: identical={same} different={diff} "
      f"missing={len(missing)} extra={len(extra)}")
import os
if diff and os.environ.get("COMPARE_REPORT_ONLY") == "1" and bad is None and not missing and not extra:
    print("  (report only: entries that a retraining step rebuilt may differ in bytes; see the step's own check)")
    sys.exit(0)
sys.exit(0 if bad is None and diff == 0 and not missing and not extra else 1)
PY
sha256sum "$OUT"
