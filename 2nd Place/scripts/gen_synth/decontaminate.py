#!/usr/bin/env python
r"""Stage 3b: remove real-transcript text copied verbatim from the few-shot exemplar.

Despite the instruction, ~2% of generated sessions carry runs of 25+ words from their exemplar.
Each clean.jsonl record names its exemplar (exemplar_src), so every session is compared only
with that one source. Lines in a long verbatim run are dropped (repair rather than regeneration,
which would reuse the same deterministic exemplar); a session left below --min-lines is removed.
Natural n-gram collisions (times tables, counting runs) are shorter than --min-run (20 words).

    python scripts/gen_synth/decontaminate.py --clean outputs/synth/clean.jsonl \
        --transcripts-dir data/train_transcripts
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NGRAM = 12  # membership grain; runs are maximal chains of consecutive hits


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def ngram_set(w: list[str], n: int) -> set[str]:
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def verbatim_line_idx(lines, real_words, min_run: int) -> set[int]:
    """Indices of lines that overlap a >=min_run-word verbatim run vs real_words."""
    real = ngram_set(real_words, NGRAM)
    w, line_of = [], []
    for li, (_, text) in enumerate(lines):
        for tok in words(text):
            w.append(tok)
            line_of.append(li)
    bad_lines: set[int] = set()
    i, limit = 0, len(w) - NGRAM + 1
    while i < limit:
        if " ".join(w[i:i + NGRAM]) in real:
            j = i
            while j < limit and " ".join(w[j:j + NGRAM]) in real:
                j += 1
            run_len = j - i + NGRAM - 1
            if run_len >= min_run:
                bad_lines.update(line_of[i:j + NGRAM - 1])
            i = j + NGRAM - 1
        else:
            i += 1
    return bad_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="outputs/synth500/clean.jsonl")
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--min-run", type=int, default=20,
                    help="verbatim word-run length that counts as copying")
    ap.add_argument("--min-lines", type=int, default=80,
                    help="drop the whole session if repair leaves fewer lines")
    args = ap.parse_args()
    tdir = Path(args.transcripts_dir).expanduser()
    clean = Path(args.clean).expanduser()

    recs = [json.loads(ln) for ln in open(clean, encoding="utf-8")]
    n_repaired = n_dropped = n_lines_cut = n_unchecked = 0
    out = []
    for rec in recs:
        src = rec.get("exemplar_src")
        if not src:
            n_unchecked += 1  # pre-recording generation: nothing to check against
            out.append(rec)
            continue
        src_path = tdir / f"{src}.csv"
        with open(src_path, newline="", encoding="utf-8", errors="replace") as f:
            real_words = words(" ".join((r.get("content") or "")
                                        for r in csv.DictReader(f)))
        bad = verbatim_line_idx(rec["lines"], real_words, args.min_run)
        if bad:
            kept = [l for i, l in enumerate(rec["lines"]) if i not in bad]
            n_lines_cut += len(bad)
            if len(kept) < args.min_lines:
                n_dropped += 1
                print(f"[decon] DROP {rec['session_id']} <- {src}: "
                      f"{len(bad)} copied lines, {len(kept)} left")
                continue
            n_repaired += 1
            print(f"[decon] repair {rec['session_id']} <- {src}: "
                  f"cut {len(bad)} copied lines")
            rec = {**rec, "lines": kept}
        out.append(rec)

    tmp = clean.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(clean)
    print(f"[decon] {len(recs)} sessions: {n_repaired} repaired "
          f"({n_lines_cut} lines cut), {n_dropped} dropped, "
          f"{n_unchecked} without exemplar_src (skipped) -> {clean}")


if __name__ == "__main__":
    main()
