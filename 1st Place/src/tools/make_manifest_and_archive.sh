#!/usr/bin/env bash
# Regenerate MANIFEST.sha256 and, optionally, pack the release archives (two .zip files) from
# exactly the files the manifests list. Never zip the folder itself: build/workspace/data/ links or
# holds the competition data.
#
#   bash src/tools/make_manifest_and_archive.sh          # rewrite the manifests (make manifest)
#   bash src/tools/make_manifest_and_archive.sh --list   # print the files they would list; writes nothing
#   bash src/tools/make_manifest_and_archive.sh --check  # compare those lists with the manifests;
#                                                        # writes nothing (make manifest-check)
#   bash src/tools/make_manifest_and_archive.sh ../trace-the-ace_prize-winner.zip   # (make archive)
#        rewrites the manifests, packs the two archives, unpacks them into an empty folder, runs
#        `sha256sum -c` there, then writes each archive's sha256 to a .sha256 file next to it
#
# Two archives, two manifests:
#   trace-the-ace_prize-winner.zip                 the repository; MANIFEST.sha256
#   trace-the-ace_prize-winner_seed-replicates.zip optional: models/llm/seeds/ (the twelve seed-replicate
#                                                  adapters, 2.1 GB); models/llm/seeds/MANIFEST.sha256
# Both unpack into the same folder trace-the-ace_prize-winner/. Without models/llm/seeds/ the script
# writes and checks MANIFEST.sha256 only, and packs only the first archive.
#
# Which files: the top-level entries in SHIP below and everything under them, except
# models/huggingface_models/ (the base model, downloaded separately) and data/raw/ (except its
# README). A manifest never lists itself. Any other top-level entry stops the script, apart from
# build/, .git/, virtual environments (.venv*) and MANIFEST.sha256, which are never shipped.
#
# The script also stops, before it writes anything, when data/processed/ holds anything besides
# .gitkeep (make predict and make example write test-set predictions there), or when a shipped
# path (data/raw/ itself included, the competition files inside it excluded) is a symbolic link or
# contains a backslash or a newline. When writing (manifest and archive, not --list or --check) it
# also stops when a __pycache__ folder exists (run make clean-cache) or when a shipped *.md file
# still holds a release placeholder (TO FILL, LINK TO BE ADDED, [PENDING]).
#
# The archives are reproducible: entries sorted by path, one top-level folder
# trace-the-ace_prize-winner/, every timestamp set to SOURCE_DATE_EPOCH (default 315532800,
# 1980-01-01 00:00:00 UTC, the earliest time a zip can store), modes 755 for *.sh, *.py and *.bsub
# and 644 for everything else, whatever the filesystem reports. Zip64 records are written only
# where they are needed (each archive is about 2 GB on 2026-10-26, under the 4 GiB limit).
# Two runs on the same tree with the same Python and zlib give the same sha256. An archive's
# sha256 goes only into the sidecar file next to it, never into the archive, whose README is
# itself listed in the manifest.
#
# Exit status: 0 on success, non-zero otherwise (1, with an ERROR line, when a guard or a check
# stops the run).
#
# Disk: about 10 GB free next to the output (the new archives are written beside any old ones
# before they replace them) and about 5 GB in $TMPDIR for the test unpack (the shipped files were
# 4.86 GB uncompressed on 2026-10-26, 2.1 GB of them the seed replicates). Needs GNU
# coreutils/findutils, unzip, and Python 3.8 or newer ($PYTHON, default python3) for the zip routine.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }
[ $# -le 1 ] || die "at most one argument"
MODE=manifest; OUT=
case "${1:-}" in            # a relative output path is taken from the caller's folder
  "") ;;
  --list) MODE=list ;;
  --check) MODE=check ;;
  -*) die "unknown option $1 (use --list, --check or an output path ending in .zip)" ;;
  *) MODE=archive; OUT=$(realpath -m "$1"); OUT_SEEDS="${OUT%.zip}_seed-replicates.zip" ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=$(pwd -P)
PREFIX=trace-the-ace_prize-winner
SHIP=(.gitattributes .gitignore LICENSE Makefile README.md THIRD_PARTY_NOTICES.md
      requirements.txt data models reports src)
SEEDS=models/llm/seeds                       # shipped in the second archive, with its own manifest
SEEDS_MANIFEST=$SEEDS/MANIFEST.sha256
PY=${PYTHON:-python3}

# --- guards (read-only, every mode) ---------------------------------------------------------------
# 1. allow-list of top-level entries: every shipped one must exist, nothing unknown may be present
for e in "${SHIP[@]}"; do [ -e "$e" ] || die "shipped top-level entry '$e' is missing"; done
stray=()
for e in .[!.]* ..?* *; do
  [ -e "$e" ] || [ -L "$e" ] || continue          # an unmatched glob stays literal
  case " ${SHIP[*]} " in *" $e "*) continue ;; esac
  case "$e" in build|.git|.venv*|MANIFEST.sha256) continue ;; esac
  stray+=("$e")
done
[ ${#stray[@]} -eq 0 ] || die "unexpected top-level entries (move or delete them): ${stray[*]}"
# 2. no test-set predictions left in data/processed/
extra=$(find data/processed -mindepth 1 -not -path data/processed/.gitkeep -print)
[ -z "$extra" ] || die "data/processed/ must hold only .gitkeep; delete these (written by make predict / make example):
$extra"
# 3. no symbolic links among the shipped paths (find -type f below would skip them silently)
# (data/raw/ itself is checked: were it a link, find would not descend and its README would drop
# out; the competition files inside it never ship, so they may be links)
links=$(find "${SHIP[@]}" \( -path models/huggingface_models -o \( -path 'data/raw/*' ! -path data/raw/README.md \) \) -prune -o -type l -print)
[ -z "$links" ] || die "symbolic links among the shipped paths (replace them by the files; data/raw/ itself must be a real folder, though the competition files inside it may be links):
$links"
# 4. sha256sum escapes names with a backslash or a newline, which `cut -c67-` would then misread
n_lines=$(find "${SHIP[@]}" -path models/huggingface_models -prune -o -type f -print | wc -l)
n_names=$(find "${SHIP[@]}" -path models/huggingface_models -prune -o -type f -print0 | tr -cd '\0' | wc -c)
[ "$n_lines" -eq "$n_names" ] || die "a shipped path contains a newline"

FILES=$(find "${SHIP[@]}" \( -path models/huggingface_models -o -path "$SEEDS" \) -prune -o -type f \
             \( -not -path 'data/raw/*' -o -path data/raw/README.md \) -print | LC_ALL=C sort)
[ -n "$FILES" ] || die "no files selected"
SEED_FILES=
if [ -d "$SEEDS" ]; then
  SEED_FILES=$(find "$SEEDS" -type f ! -path "$SEEDS_MANIFEST" -print | LC_ALL=C sort)
fi
if grep -qF '\' <<<"$FILES$SEED_FILES"; then die "a shipped path contains a backslash"; fi

if [ "$MODE" = list ]; then printf '%s\n' "$FILES"; [ -z "$SEED_FILES" ] || printf '%s\n' "$SEED_FILES"; exit 0; fi

# check_cover MANIFEST LIST: does the manifest name exactly the files in LIST?
check_cover() {
  [ -f "$1" ] || die "$1 is missing"
  local d
  d=$(diff <(cut -c67- "$1" | LC_ALL=C sort) <(printf '%s\n' "$2") || true)
  if [ -n "$d" ]; then
    sed -n -e 's/^> /  not in the manifest: /p' -e 's/^< /  in the manifest but not present: /p' <<<"$d" >&2
    die "$1 does not cover exactly the shipped files (above); run make manifest"
  fi
  echo "$1 lists exactly the $(wc -l <<<"$2") shipped files (contents: make verify-repo)"
}

if [ "$MODE" = check ]; then   # coverage: the selection against the names the manifests list
  check_cover MANIFEST.sha256 "$FILES"
  [ -z "$SEED_FILES" ] || check_cover "$SEEDS_MANIFEST" "$SEED_FILES"
  exit 0
fi

# --- guards that apply only when writing ----------------------------------------------------------
# 5. no __pycache__ (the virtual environments, the base model and build/ are not looked at)
pyc=$(find . \( -path ./build -o -path ./.git -o -path './.venv*' -o -path ./models/huggingface_models \) \
        -prune -o -name __pycache__ -print)
[ -z "$pyc" ] || die "__pycache__ folders present; run 'make clean-cache' first:
$pyc"
# 6. no release placeholders left in any shipped Markdown file (README.md, ...)
mapfile -t MD < <(grep -E '\.md$' <<<"$FILES$([ -z "$SEED_FILES" ] || printf '\n%s' "$SEED_FILES")")
[ ${#MD[@]} -gt 0 ] || die "no Markdown files selected"
rc=0; grep -nHF -e 'TO FILL' -e 'LINK TO BE ADDED' -e '[PENDING]' -- "${MD[@]}" >&2 || rc=$?
[ "$rc" -ne 0 ] || die "release placeholders left in the Markdown files (above)"
[ "$rc" -eq 1 ] || die "grep failed (exit $rc) while looking for release placeholders"

if [ "$MODE" = archive ]; then   # check the output path and tools before writing the manifest
  case "$OUT" in "$ROOT"|"$ROOT"/*) die "write the archive outside the repository" ;; esac
  case "$OUT" in *.zip) ;; *) die "the archive name must end in .zip" ;; esac
  command -v unzip >/dev/null || die "unzip is needed to test the archive"
  "$PY" -c 'import sys, zipfile; sys.exit(sys.version_info < (3, 8))' \
    || die "\$PYTHON ($PY) must be Python 3.8 or newer"
  EPOCH=${SOURCE_DATE_EPOCH:-315532800}
  [[ "$EPOCH" =~ ^[0-9]+$ ]] || die "SOURCE_DATE_EPOCH must be a whole number of seconds"
fi

printf '%s\n' "$FILES" | xargs -d '\n' sha256sum > MANIFEST.sha256
echo "MANIFEST.sha256: $(wc -l < MANIFEST.sha256) files"
sha256sum -c --quiet MANIFEST.sha256 || die "the repository does not match the manifest just written"
echo "REPOSITORY INTACT"
if [ -n "$SEED_FILES" ]; then
  printf '%s\n' "$SEED_FILES" | xargs -d '\n' sha256sum > "$SEEDS_MANIFEST"
  echo "$SEEDS_MANIFEST: $(wc -l < "$SEEDS_MANIFEST") files"
  sha256sum -c --quiet "$SEEDS_MANIFEST" || die "$SEEDS/ does not match the manifest just written"
  echo "SEED REPLICATES INTACT"
else
  echo "no $SEEDS/: only MANIFEST.sha256 written"
fi
[ "$MODE" = archive ] || exit 0

# --- archive -------------------------------------------------------------------------------------
# Deterministic zip: entries sorted by path (folders included), one top-level folder, fixed
# timestamps, fixed Unix modes, deflate (stored for files that are already compressed).
ZIPPY=$(cat <<'PY'
import os, shutil, stat, sys, time, zipfile
root, out, prefix, epoch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
names = [n for n in sys.stdin.read().split("\n") if n]
date_time = time.gmtime(max(epoch, 315532800))[:6]         # a zip cannot store times before 1980
entries = {prefix + "/": None}
for n in names:
    parts = n.split("/")
    for i in range(1, len(parts)):
        entries[prefix + "/" + "/".join(parts[:i]) + "/"] = None
    entries[prefix + "/" + n] = os.path.join(root, n)
with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
    for arc in sorted(entries, key=lambda s: s.encode("utf-8")):
        src = entries[arc]
        zi = zipfile.ZipInfo(arc, date_time)
        zi.create_system = 3                                    # Unix, so unzip applies the modes
        if src is None:
            zi.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
            zf.writestr(zi, b"")
            continue
        mode = 0o755 if arc.endswith((".sh", ".py", ".bsub")) else 0o644
        zi.external_attr = (stat.S_IFREG | mode) << 16
        zi.compress_type = zipfile.ZIP_STORED if arc.endswith((".zip", ".gz")) else zipfile.ZIP_DEFLATED
        zi.file_size = os.path.getsize(src)                     # lets zipfile choose Zip64 per entry
        with open(src, "rb") as f, zf.open(zi, "w") as dst:
            shutil.copyfileobj(f, dst, 1 << 20)
print(f"packed {len(names)} files into {out}")
PY
)
mkdir -p "$(dirname "$OUT")"
PART="$OUT.partial"; PART_SEEDS=; T=
trap 'rm -rf "${T:-}" "${PART:-}" "${PART_SEEDS:-}"' EXIT
rm -f "$PART" "$OUT.sha256" "$OUT_SEEDS.partial" "$OUT_SEEDS.sha256"
{ cut -c67- MANIFEST.sha256; echo MANIFEST.sha256; } | "$PY" -I -c "$ZIPPY" "$ROOT" "$PART" "$PREFIX" "$EPOCH"
if [ -n "$SEED_FILES" ]; then
  PART_SEEDS="$OUT_SEEDS.partial"
  { cut -c67- "$SEEDS_MANIFEST"; echo "$SEEDS_MANIFEST"; } \
    | "$PY" -I -c "$ZIPPY" "$ROOT" "$PART_SEEDS" "$PREFIX" "$EPOCH"
fi

# every member under the one top-level folder, none from the paths that must never ship, and no
# file in both archives
for z in "$PART" ${PART_SEEDS:+"$PART_SEEDS"}; do
  members=$(unzip -Z1 "$z")
  bad=$(grep -v "^$PREFIX/" <<<"$members" || true)
  [ -z "$bad" ] || die "members outside $PREFIX/ in $(basename "$z"):
$bad"
  bad=$(grep -E "^$PREFIX/(build/|\.git/|\.venv|models/huggingface_models/|data/raw/.|data/processed/.)|/__pycache__/" <<<"$members" \
          | grep -vxF -e "$PREFIX/data/raw/README.md" -e "$PREFIX/data/processed/.gitkeep" || true)
  [ -z "$bad" ] || die "$(basename "$z") contains paths it must not:
$bad"
done
if [ -n "$PART_SEEDS" ]; then
  both=$(LC_ALL=C comm -12 <(unzip -Z1 "$PART" | grep -v '/$' | LC_ALL=C sort) \
                  <(unzip -Z1 "$PART_SEEDS" | grep -v '/$' | LC_ALL=C sort))
  [ -z "$both" ] || die "files in both archives:
$both"
fi

# unpack into an empty folder and check it there, as a recipient would: the repository on its own
# first, then the seed replicates into the same folder
unpacked() { (cd "$T/$PREFIX" && find . -type f | sed 's|^\./||' | LC_ALL=C sort); }
T=$(mktemp -d)
unzip -q "$PART" -d "$T" || die "unzip failed"
[ "$(ls -A "$T")" = "$PREFIX" ] || die "the archive does not unpack into the one folder $PREFIX/"
( cd "$T/$PREFIX" && sha256sum -c --quiet MANIFEST.sha256 ) || die "the archive does not unpack intact"
expect=$({ cut -c67- MANIFEST.sha256; echo MANIFEST.sha256; } | LC_ALL=C sort)
d=$(diff <(printf '%s\n' "$expect") <(unpacked) || true)
[ -z "$d" ] || die "the unpacked files differ from the manifest list:
$d"
echo "archive unpacks INTACT"
if [ -n "$PART_SEEDS" ]; then
  unzip -q -n "$PART_SEEDS" -d "$T" || die "unzip of the seed-replicates archive failed"
  [ "$(ls -A "$T")" = "$PREFIX" ] || die "the seed-replicates archive does not unpack into $PREFIX/"
  ( cd "$T/$PREFIX" && sha256sum -c --quiet "$SEEDS_MANIFEST" ) || die "the seed replicates do not unpack intact"
  expect=$({ printf '%s\n' "$expect"; cut -c67- "$SEEDS_MANIFEST"; echo "$SEEDS_MANIFEST"; } | LC_ALL=C sort)
  d=$(diff <(printf '%s\n' "$expect") <(unpacked) || true)
  [ -z "$d" ] || die "the unpacked files differ from the two manifest lists:
$d"
  echo "seed-replicates archive unpacks INTACT into the same folder"
fi
noexec=$(cd "$T/$PREFIX" && find . -type f \( -name '*.sh' -o -name '*.py' -o -name '*.bsub' \) ! -perm -u+x -print)
[ -z "$noexec" ] || die "scripts unpacked without the executable bit:
$noexec"

# publish ZIP: move the tested .partial into place and write its .sha256 next to it
publish() {
  mv -f "$1.partial" "$1"
  ( cd "$(dirname "$1")" && sha256sum "$(basename "$1")" > "$(basename "$1").sha256" )
  cat "$1.sha256"
}
publish "$OUT"; PART=
if [ -n "$PART_SEEDS" ]; then publish "$OUT_SEEDS"; PART_SEEDS=; fi
echo "publish each .sha256 (or its line, in the e-mail) next to its archive, never inside it"
