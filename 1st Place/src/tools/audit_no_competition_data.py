#!/usr/bin/env python
"""Check that this repository contains no competition data, and print exactly what was checked.

    python src/tools/audit_no_competition_data.py --raw data/raw

``--raw`` is the folder with DrivenData's files; each input is found by its name prefix in that
folder or in its ``test/`` subfolder. Required: ``train_features*``, ``train_labels*``, the train
transcripts (folder ``train_transcripts/`` or a ``*transcripts*.zip``), ``test_features*``,
``test_labels*`` and the test transcripts. Optional: ``qualitative_metadata*``. If a required input
is missing, the tool still scans with what it has, prints ``RESULT: INCOMPLETE`` and exits with
status 2.

What is indexed (all text lowercased and split into tokens ``[a-z0-9']+``; a token is a word if it
contains a letter):

1. long quotes: every run of 12 consecutive tokens (at least 8 of them words) of every utterance
   of the train and test transcripts. Any quote of 12 or more consecutive tokens of one utterance
   is found if 12 of them in a row include at least 8 words and that run does not also occur in a
   pinned public file (below).
2. short utterances: every utterance of 8-11 tokens (at least 7 of them words) that occurs at most
   3 times in all the transcripts, as a whole; it is found when all its tokens appear in a row.
   Frequent ones are stock phrases ("can you tell me what you got") and are not indexed.
   Utterances of fewer than 8 tokens are not indexed, and a quote of fewer than 12 tokens taken
   from inside a longer utterance is not detected.
3. objective texts (train and test), as token sequences: a text of 4 or more tokens is found
   anywhere; a text of 2-3 tokens ("simplifying fractions") only as a whole line or a whole table cell.
4. labels (train and test): in any table with a ``response_id`` column, every column whose values
   are all 0/1 is compared with the labels on the shared ids, and flagged if it agrees (or
   disagrees) on more than 95% of at least 100 shared ids. Tables without ``response_id`` and 1-D
   ``.npy`` arrays with exactly as many rows as the train or test set are compared row for row, in
   the order of the features file and of the labels file. 1-D arrays of that length inside
   ``.joblib`` files are checked the same way.
5. the ids in ``qualitative_metadata`` (student_id, tutor_id, question_id), as whole tokens.

Public wording is taken out of the index before the scan: a 12-token run, a short utterance or an
objective text that also occurs in one of the third-party texts shipped in ``data/external/``
(MRBench and the national-curriculum statements, each pinned by sha256 in ``PUBLIC_REFERENCES``)
is public text, not competition data. Tutors read curriculum statements aloud and MRBench's tutor
responses use the same classroom phrases, so without this step those files, and anything quoting
them, would match. A reference file whose sha256 differs is not used and is flagged.

What is scanned: every file under the repository except the folders that are not part of it
(``.git/``, ``build/``, ``data/raw/`` apart from its README (none of its subfolders is entered),
the ``--raw`` folder itself if it lies inside the repository, ``models/huggingface_models/``,
``.venv*/``, ``__pycache__/``):

* every file that is not binary (no NUL byte in its first 8 KB), as text, whatever its extension;
* ``.pkl``, ``.pkl.gz``, ``.parquet``: every text column (object or string dtype, distinct values),
  and the label check;
* ``.npy``: the label check;
* ``models/submission.zip``: each member is compared with the file of the same path in
  ``models/submission/``; members that differ, or have no counterpart, are scanned as text;
* ``.joblib``: 1-D label arrays (pass/fail) and an informational vocabulary report (below);
* ``.safetensors`` (LoRA weights) and other binary files are listed but not scanned.

Whitelisted: ``models/submission/assets/tokenizer/*`` is scanned and any hits are printed, but they
do not count. These are the Qwen3-8B tokenizer files (vocab.json, merges.txt and tokenizer.json are
checked against the base model's sha256); a list of 151k sub-word tokens can contain, by chance,
runs that also occur in transcripts.

Vocabulary report (informational, not part of the result): for every fitted TF-IDF vectorizer in
the ``.joblib`` files, its size, whether ``vocabulary_`` is stored in sorted order, how many of its
terms are word bigrams, how many of those bigrams occur as two consecutive words in a transcript
utterance (and in an objective text), and the longest chain of consecutive dict keys that link up
("a b", "b c", ...) in words. A vocabulary stored in first-occurrence order replays its training
text through such chains; a sorted one does not. The single words and word pairs of the training
text are in the vocabulary of any trained text model.

The output never quotes the matched text: it gives the file, the line or column, and counts.
``--show`` adds the matched lines on stderr, for the maintainers' own review; never save that
output in the repository.

Exit status: 0 passed, 1 files to review, 2 incomplete (a required input is missing).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime
import gzip
import hashlib
import io
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True           # loading the joblibs imports magnificat: leave no __pycache__

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SKIP_PREFIXES = (".git/", "build/", "data/raw/", "models/huggingface_models/")
SKIP_KEEP = {"data/raw/README.md"}       # ours, inside a skipped folder
WHITELIST = {
    "models/submission/assets/tokenizer/":
        "the Qwen3-8B tokenizer: vocab.json, merges.txt and tokenizer.json are the base model's files "
        "byte for byte (checked below against the sha256 of Qwen/Qwen3-8B at revision b968826), the "
        "other three are small configuration files written by transformers; a list of 151k sub-word "
        "tokens can contain, by chance, runs of tokens that also occur in transcripts",
}
# sha256 of the tokenizer files of Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218
BASE_TOKENIZER_SHA = {
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
}
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".gguf")

TOK = re.compile(r"[a-z0-9']+")          # tokens of the quote index
WORD = re.compile(r"(?u)\b\w\w+\b")      # scikit-learn's default token_pattern (vocabulary report)
ID_TOK = re.compile(r"[A-Za-z0-9_]+")
LONG_N, LONG_MIN_WORDS = 12, 8
SHORT_N, SHORT_MIN_WORDS, SHORT_MAX_COUNT = 8, 7, 3   # whole utterances of 8-11 tokens, >= 7 words,
                                                     # that occur at most 3 times
# Third-party public texts shipped in the repository, pinned by sha256. A run of tokens that also
# occurs in one of them is public wording (tutors read curriculum statements aloud; MRBench's
# tutor responses use the same classroom phrases), so it is removed from the index before the scan.
PUBLIC_REFERENCES = {
    "data/external/mrbench/raw/MRBench_V1.json":
        "77a33ab81ca9c773d22faee2be7a7bb43b203a06497a5daa35d715f9ada3a6f3",
    "data/external/mrbench/raw/MRBench_V2.json":
        "a8792225075533d15c23e1c903d0c649f19708ee93f5c3d8267eaa6ab623aac1",
    "data/external/mrbench/raw/mrbench_v3_devset.json":
        "a395feab801de50d44b73e1e9b88f84824a1f1d92e15ae86842d90c2b6dbf33c",
    "data/external/mrbench/raw/mrbench_v3_testset.json":
        "5a73434fd9e1dfc73026e146d3709f95f2483a51a5e5b354bc96d981448d92cd",
    "data/external/nc_curriculum.csv":
        "0f0c72bd8235a4778dfd6ffebbae95f3f1c6678342e1c48f69d7e31392d1c5fc",
}
LO_MIN_ANYWHERE = 4                      # objective texts of >= 4 tokens are matched anywhere
LABEL_AGREE, LABEL_MIN_SHARED = 0.95, 100
P = np.uint64(0x9E3779B97F4A7C15)        # multiplier of the rolling n-gram hash (mod 2**64)
BATCH = 200_000                          # utterances per vectorised batch


# --------------------------------------------------------------------------------------------
# n-gram hashing: each token gets a 64-bit hash, an n-gram is the polynomial sum mod 2**64

class Hasher:
    """Token -> (64-bit hash, is a word). Consistent within one run (Python's str hash)."""

    def __init__(self) -> None:
        self.cache: dict[str, int] = {}
        self.H = np.empty(0, np.uint64)
        self.W = np.empty(0, bool)

    def encode(self, toks: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Hash a token list.

        Args:
            toks: tokens, in order.

        Returns:
            (hashes, is_word): uint64 hash and bool "contains a letter" per token.
        """
        c = self.cache
        new = [x for x in dict.fromkeys(toks) if x not in c]
        if new:
            for x in new:
                c[x] = len(c)
            self.H = np.concatenate([self.H, np.array([hash(x) for x in new], np.int64).view(np.uint64)])
            self.W = np.concatenate([self.W, np.array([any(ch.isalpha() for ch in x) for x in new], bool)])
        ids = np.fromiter(map(c.__getitem__, toks), np.int64, len(toks))
        return self.H[ids], self.W[ids]

    def seq_hashes(self, seqs: list[tuple[str, ...]]) -> np.ndarray:
        """The n-gram hash of each sequence (all of the same length), as ``ngrams`` computes it."""
        if not seqs:
            return np.empty(0, np.uint64)
        n = len(seqs[0])
        h, _ = self.encode([x for s in seqs for x in s])
        h = h.reshape(len(seqs), n)
        acc = np.zeros(len(seqs), np.uint64)
        with np.errstate(over="ignore"):
            for k in range(n):
                acc = acc * P + h[:, k]
        return acc


def ngrams(h: np.ndarray, isword: np.ndarray, starts: np.ndarray, n: int, min_words: int):
    """Hashes and start positions of every n-gram that lies inside one segment and has enough words.

    Args:
        h: uint64 token hashes of the concatenated segments.
        isword: bool per token, True if it contains a letter.
        starts: first token index of each segment (utterance, file); n-grams never cross a
            segment boundary.
        n: n-gram length in tokens.
        min_words: minimum number of word tokens in an n-gram.

    Returns:
        (hashes, positions): the rolling hash and start token index of each kept n-gram.
    """
    m = len(h) - n + 1
    if m <= 0:
        return np.empty(0, np.uint64), np.empty(0, np.int64)
    with np.errstate(over="ignore"):
        acc = np.zeros(m, dtype=np.uint64)
        for k in range(n):
            acc = acc * P + h[k:k + m]
    seg = np.zeros(len(h), dtype=np.int64)
    seg[starts[(starts > 0) & (starts < len(h))]] = 1      # empty segments share a start: harmless
    seg = np.cumsum(seg)
    cs = np.concatenate([[0], np.cumsum(isword, dtype=np.int64)])
    pos = np.arange(m)
    ok = (seg[pos] == seg[pos + n - 1]) & (cs[pos + n] - cs[pos] >= min_words)
    return acc[ok], pos[ok]


def member(idx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``np.isin(x, idx)`` for a sorted, unique ``idx`` (no re-sort of the large index)."""
    if not len(idx) or not len(x):
        return np.zeros(len(x), bool)
    i = np.searchsorted(idx, x)
    i[i == len(idx)] = 0
    return idx[i] == x


def lo_matches(toks: list[str], h, w, prefix: dict[int, list[tuple]]):
    """(objective token tuple, token position) for every long objective text found in ``toks``."""
    if not prefix:
        return []
    hh, pos = ngrams(h, w, np.zeros(1, np.int64), LO_MIN_ANYWHERE, 0)
    keys = np.fromiter(prefix, dtype=np.uint64, count=len(prefix))
    sel = np.isin(hh, keys)
    out = []
    for hv, p in zip(hh[sel], pos[sel]):
        for lo in prefix[int(hv)]:
            if tuple(toks[p:p + len(lo)]) == lo:
                out.append((lo, int(p)))
    return out


def whole_utterances(h, w, starts, lens):
    """Hashes of the utterances of SHORT_N..LONG_N-1 tokens with >= SHORT_MIN_WORDS words."""
    out = []
    for n in range(SHORT_N, LONG_N):
        sel = lens == n
        if not sel.any():
            continue
        ix = starts[sel][:, None] + np.arange(n)
        ok = w[ix].sum(1) >= SHORT_MIN_WORDS
        hm = h[ix[ok]]
        acc = np.zeros(len(hm), np.uint64)
        with np.errstate(over="ignore"):
            for k in range(n):
                acc = acc * P + hm[:, k]
        out.append(acc)
    return np.concatenate(out) if out else np.empty(0, np.uint64)


def short_runs(h, w, starts):
    """Hashes and positions of every SHORT_N..LONG_N-1 token run (for matching whole utterances)."""
    hs, ps = zip(*(ngrams(h, w, starts, n, SHORT_MIN_WORDS) for n in range(SHORT_N, LONG_N)))
    return np.concatenate(hs), np.concatenate(ps)


# --------------------------------------------------------------------------------------------
# inputs

def find_inputs(raw: Path) -> dict[str, list[Path]]:
    """Locate the competition files in ``raw`` and ``raw/test`` by their name prefixes.

    Args:
        raw: the folder with DrivenData's files.

    Returns:
        {input name: files found}, for train/test features, labels and transcripts (a folder or
        zips) and qualitative_metadata; an empty list where nothing was found.
    """
    dirs = [raw, raw / "test"]

    def g(pattern):
        return sorted({p for d in dirs for p in d.glob(pattern) if p.is_file()})
    inp = {
        "train_features": g("train_features*.csv"),
        "train_labels": g("train_labels*.csv"),
        "test_features": g("test_features*.csv"),
        "test_labels": g("test_labels*.csv"),
        "qualitative_metadata": g("qualitative_metadata*.csv"),
    }
    for kind in ("train", "test"):
        folders = [d / f"{kind}_transcripts" for d in dirs
                   if (d / f"{kind}_transcripts").is_dir() and any((d / f"{kind}_transcripts").glob("*.csv"))]
        zips = [p for p in g("*transcripts*.zip") if (kind == "test") == ("test" in p.name)]
        inp[f"{kind}_transcripts"] = folders[:1] or zips
    return inp


def iter_utterances(src: Path):
    """Yield the utterance texts of a transcripts folder or zip."""
    if src.is_dir():
        for f in sorted(src.glob("*.csv")):
            yield from pd.read_csv(f, usecols=["content"]).content.dropna().astype(str)
        return
    with zipfile.ZipFile(src) as zf:
        for m in zf.infolist():
            if not m.filename.endswith(".csv"):
                continue
            try:
                payload = zf.open(m).read()
            except zipfile.BadZipFile:                     # a member with a wrong stored CRC
                f = zf.open(m); f._expected_crc = None; payload = f.read()
            yield from pd.read_csv(io.BytesIO(payload), usecols=["content"]).content.dropna().astype(str)


def build_index(sources: list[Path], hasher: Hasher, vocab_bigrams: np.ndarray):
    """Index the transcripts: 12-token runs and rare whole short utterances.

    Args:
        sources: transcript folders or zips.
        hasher: the run's token hasher (hashes are only comparable within one run).
        vocab_bigrams: sorted hashes of the word bigrams of every vectorizer vocabulary, for the
            informational vocabulary report.

    Returns:
        (long_idx, short_idx, stats, seen_bigram): sorted unique hashes of the 12-token runs,
        sorted hashes of the 8-11-token utterances that occur at most 3 times, utterance counts
        by length, and a bool mask over ``vocab_bigrams`` of those seen in a transcript.
    """
    long_parts, short_parts = [], []
    seen_bigram = np.zeros(len(vocab_bigrams), dtype=bool)
    st = defaultdict(int)
    batch_q: list[list[str]] = []
    batch_w: list[list[str]] = []

    def flush():
        if not batch_q:
            return
        lens = np.fromiter((len(t) for t in batch_q), dtype=np.int64, count=len(batch_q))
        flat = [x for t in batch_q for x in t]
        h, w = hasher.encode(flat)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        long_parts.append(ngrams(h, w, starts, LONG_N, LONG_MIN_WORDS)[0])
        short_parts.append(whole_utterances(h, w, starts, lens))
        if len(vocab_bigrams):
            wl = np.fromiter((len(t) for t in batch_w), dtype=np.int64, count=len(batch_w))
            hw, ww = hasher.encode([x for t in batch_w for x in t])
            bg = ngrams(hw, np.ones(len(hw), bool), np.concatenate([[0], np.cumsum(wl)[:-1]]), 2, 0)[0]
            seen_bigram[np.isin(vocab_bigrams, bg)] = True
        batch_q.clear(); batch_w.clear()

    for src in sources:
        for u in iter_utterances(src):
            lo = u.lower()
            t = TOK.findall(lo)
            st["utterances"] += 1
            n = len(t)
            st["tokens >= 12"] += n >= LONG_N
            st["tokens 8-11"] += SHORT_N <= n < LONG_N
            st["tokens < 8"] += n < SHORT_N
            batch_q.append(t)
            if len(vocab_bigrams):
                batch_w.append(WORD.findall(lo))
            if len(batch_q) >= BATCH:
                flush()
    flush()
    long_idx = np.unique(np.concatenate(long_parts)) if long_parts else np.empty(0, np.uint64)
    allshort = np.concatenate(short_parts) if short_parts else np.empty(0, np.uint64)
    del short_parts
    keys, counts = np.unique(allshort, return_counts=True)
    del allshort
    st["short utterances (distinct)"] = len(keys)
    short_idx = keys[counts <= SHORT_MAX_COUNT]
    return long_idx, short_idx, dict(st), seen_bigram


def load_objectives(paths: list[Path], hasher: Hasher):
    """Load the objective texts as token tuples (ids are printed, never the texts).

    Args:
        paths: features files with learning_objective_id and learning_objective.
        hasher: the run's token hasher.

    Returns:
        (long_los, short_los, prefix): {tokens: id} for texts of 4 or more tokens and of fewer,
        and {hash of the first 4 tokens: [long texts]} for matching long texts anywhere.
    """
    los: dict[tuple, str] = {}
    for f in paths:
        df = pd.read_csv(f, usecols=["learning_objective_id", "learning_objective"])
        for i, t in zip(df.learning_objective_id.astype(str), df.learning_objective.fillna("").astype(str)):
            toks = tuple(TOK.findall(t.lower()))
            if toks:
                los.setdefault(toks, i)
    long_los = {k: v for k, v in los.items() if len(k) >= LO_MIN_ANYWHERE}
    short_los = {k: v for k, v in los.items() if len(k) < LO_MIN_ANYWHERE}
    prefix: dict[int, list[tuple]] = defaultdict(list)
    keys = list(long_los)
    for k, h in zip(keys, hasher.seq_hashes([k[:LO_MIN_ANYWHERE] for k in keys])):
        prefix[int(h)].append(k)
    return long_los, short_los, prefix


def load_labels(feat: list[Path], lab: list[Path]):
    """Load the labels of one split.

    Args:
        feat: features files of the split (for their row order); may be empty.
        lab: labels files of the split; if empty, nothing is loaded.

    Returns:
        ({response_id: 0.0/1.0}, [(order name, label vector)]), with one vector in labels-file
        row order and, if features files are given, one in features-file row order.
    """
    if not lab:
        return {}, []
    df = pd.concat([pd.read_csv(f) for f in lab])
    d = dict(zip(df.response_id.astype(str), df.is_correct.astype(float)))
    orders = [("labels-file order", df.is_correct.to_numpy(float))]
    if feat:
        ids = pd.concat([pd.read_csv(f, usecols=["response_id"]) for f in feat]).response_id.astype(str)
        orders.append(("features-file order", np.array([d.get(i, np.nan) for i in ids], float)))
    return d, orders


# --------------------------------------------------------------------------------------------
# scanning

class Scanner:
    """Matches file contents against the transcript, objective, label and id indexes.

    Built from a dict of attributes: ``hasher``, ``long_idx``, ``short_idx``, ``long_los``,
    ``short_los``, ``lo_prefix``, ``labels``, ``orders`` and ``ids`` (see ``main``). Every check
    appends a one-line description of each hit to the caller's ``hits`` list.
    """

    def __init__(self, a) -> None:
        self.__dict__.update(a)

    def scan_text(self, text: str, where: str, hits: list, show: list, cells: bool = False,
                  lines: bool = True) -> None:
        """Look for transcript runs, objective texts and metadata ids in one text.

        Args:
            text: the text to scan.
            where: prefix naming the column or zip member in the hit message ("" for a file).
            hits: list the hit descriptions are appended to.
            show: list of (line, run length) pairs appended for ``--show``.
            cells: also split delimited text into cells when matching short objective texts.
            lines: False for table columns, whose "lines" are distinct cell values, so only the
                column is reported.
        """
        low = text.lower()
        ms = list(TOK.finditer(low))
        if ms:
            toks = [m.group() for m in ms]
            h, w = self.hasher.encode(toks)
            nl = [m.start() for m in re.finditer("\n", low)]

            def line(tok_i):
                return bisect.bisect_left(nl, ms[tok_i].start()) + 1
            z = np.zeros(1, np.int64)
            for kind, idx, label in (("long", self.long_idx, f"{LONG_N}-token runs"),
                                     ("short", self.short_idx, "whole short utterances")):
                hh, pos = (ngrams(h, w, z, LONG_N, LONG_MIN_WORDS) if kind == "long"
                           else short_runs(h, w, z))
                n = LONG_N if kind == "long" else SHORT_N
                hit = pos[member(idx, hh)]
                if len(hit):
                    lns = sorted({line(int(p)) for p in hit})
                    hits.append(f"{len(hit)} {label} from transcripts ({where}"
                                + (f"lines {_fmt(lns)})" if lines else f"{len(lns)} values)"))
                    show.extend((ln, n) for ln in lns[:20])
            if self.lo_prefix:
                found = {}
                for lo, p in lo_matches(toks, h, w, self.lo_prefix):
                    found.setdefault(self.long_los[lo], line(p))
                if found:
                    first, ln = next(iter(found.items()))
                    hits.append(f"objective text of {len(found)} objective(s) ({where}e.g. id {first}"
                                + (f" at line {ln})" if lines else ")"))
            if self.ids:
                ids = set(ID_TOK.findall(text)) & self.ids
                if ids:
                    hits.append(f"{len(ids)} qualitative_metadata id(s)"
                                + (f" ({where.rstrip(', ')})" if where else ""))
        # short objective texts: whole lines, and whole cells of delimited text
        if self.short_los:
            found = {}
            units = text.splitlines()
            if cells:
                try:
                    units = units + [c for row in csv.reader(io.StringIO(text)) for c in row]
                except csv.Error:
                    pass
            for u in units:
                k = tuple(TOK.findall(u.lower()))
                if k in self.short_los:
                    found[self.short_los[k]] = 1
            if found:
                hits.append(f"short objective text as a whole line/cell: {len(found)} objective(s) "
                            f"({where}e.g. id {next(iter(found))})")

    def check_label_columns(self, df: pd.DataFrame, hits: list) -> None:
        """Flag 0/1 columns of a table that agree (or disagree) with the labels on > 95% of rows.

        Columns are matched by response_id when the table has one, otherwise row for row
        against every label vector of the same length. Hits are appended to ``hits``.
        """
        n = len(df)
        has_id = "response_id" in df.columns
        if not has_id and not any(len(y) == n for o in self.orders.values() for _, y in o):
            return
        ids = df["response_id"].astype(str) if has_id else None
        for c in df.columns:
            if c == "response_id":
                continue
            col = df[c]
            if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
                continue
            v = pd.to_numeric(col, errors="coerce").to_numpy(float)
            ok = ~np.isnan(v)
            if ok.sum() < LABEL_MIN_SHARED or not np.isin(v[ok], (0.0, 1.0)).all():
                continue
            for name, labels in self.labels.items():
                if has_id:
                    y = np.array([labels.get(i, np.nan) for i in ids], float)
                    self._agree(v, y, f"column {c!r} vs {name} labels by response_id", hits)
                else:
                    for order, y in self.orders[name]:
                        if len(y) == n:
                            self._agree(v, y, f"column {c!r} vs {name} labels row for row ({order})", hits)

    def check_array(self, a: np.ndarray, what: str, hits: list) -> None:
        """Flag a 1-D 0/1 array that matches a label vector of the same length row for row."""
        if a.ndim != 1 or a.dtype.kind not in "biuf":
            return
        v = a.astype(float)
        if not np.isin(v[~np.isnan(v)], (0.0, 1.0)).all():
            return
        for name in self.labels:
            for order, y in self.orders[name]:
                if len(y) == len(v):
                    self._agree(v, y, f"{what} vs {name} labels row for row ({order})", hits)

    @staticmethod
    def _agree(v, y, what, hits):
        ok = ~np.isnan(v) & ~np.isnan(y)
        if ok.sum() < LABEL_MIN_SHARED:
            return
        agree = float((v[ok] == y[ok]).mean())
        if agree > LABEL_AGREE or agree < 1 - LABEL_AGREE:
            hits.append(f"{what}: agrees on {agree:.1%} of {int(ok.sum()):,} rows")


def _fmt(lines: list[int]) -> str:
    return ", ".join(map(str, lines[:8])) + (f" and {len(lines) - 8} more" if len(lines) > 8 else "")


def is_binary(data: bytes) -> bool:
    """True if the first 8 KB contain a NUL byte."""
    return b"\0" in data[:8192]


def table_texts(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(column name, its distinct values joined by newlines) for every text column of a table."""
    out = []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_string_dtype(col) or col.dtype == object:
            vals = pd.unique(col.dropna().astype(str))
            out.append((str(c), "\n".join(vals)))
    return out


# --------------------------------------------------------------------------------------------
# joblib walk: label arrays (pass/fail) and vectorizer vocabularies (informational)

def walk_joblib(obj):
    """Find the fitted vectorizers and 1-D numeric arrays inside a loaded joblib object.

    Args:
        obj: the loaded object; dicts, lists, tuples and object attributes are walked, at most
            40 levels deep.

    Returns:
        (vectorizers, arrays): lists of (attribute path, object) pairs.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    vecs, arrays, seen = [], [], set()

    def rec(o, path, depth):
        if id(o) in seen or depth > 40:
            return
        seen.add(id(o))
        if isinstance(o, CountVectorizer):
            if getattr(o, "vocabulary_", None):
                vecs.append((path, o))
            return
        if isinstance(o, np.ndarray):
            if o.ndim == 1 and o.dtype.kind in "biuf":
                arrays.append((path, o))
            return
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(o, dict):
            items = [(f"[{k!r}]", v) for k, v in o.items()]
        elif isinstance(o, (list, tuple)):
            items = [(f"[{i}]", v) for i, v in enumerate(o)]
        elif hasattr(o, "__dict__") and not isinstance(o, type):
            items = [(f".{k}", v) for k, v in vars(o).items()]
        else:
            return
        for k, v in items:
            rec(v, path + k, depth + 1)
    rec(obj, "", 0)
    return vecs, arrays


def vocab_stats(vec, hasher: Hasher):
    """Summarise one vectorizer vocabulary for the vocabulary report.

    Args:
        vec: a fitted CountVectorizer or TfidfVectorizer.
        hasher: the run's token hasher.

    Returns:
        A dict with the vocabulary size (``n``), whether it is stored sorted, its word bigrams
        and their hashes, and the longest chain of consecutive keys "a b", "b c", ... in words.
    """
    keys = list(vec.vocabulary_)
    is_sorted = all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
    bigrams = [k for k in keys if k.count(" ") == 1] if vec.analyzer == "word" else []
    # longest chain of consecutive dict keys "a b", "b c", ... (in words)
    best = run = 0
    prev = None
    for k in keys:
        parts = k.split(" ")
        if len(parts) == 2 and prev is not None and prev[1] == parts[0]:
            run += 1
        else:
            run = 1 if len(parts) == 2 else 0
        best = max(best, run)
        prev = parts if len(parts) == 2 else None
    return {"n": len(keys), "sorted": is_sorted, "bigrams": bigrams,
            "bigram_hashes": hasher.seq_hashes([tuple(b.split(" ")) for b in bigrams]),
            "chain_words": best + 1 if best else 0}


# --------------------------------------------------------------------------------------------

def main() -> None:
    """Index the competition data, scan every repository file and print what was found.

    Returns:
        Nothing; prints the inputs, index sizes, scan counts, whitelist hits, the vocabulary
        report and the result, and exits 0 (passed), 1 (files to review) or 2 (a required
        input is missing). Writes no files.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="folder with the competition files")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--show", action="store_true",
                    help="print matched lines on stderr (for review only; do not save in the repository)")
    args = ap.parse_args()
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo / "src/training/src"), str(repo / "models/submission/assets")]
    hasher = Hasher()

    print(f"# python src/tools/audit_no_competition_data.py --raw <competition files>   "
          f"({datetime.date.today().isoformat()})")

    # ---- inputs ---------------------------------------------------------------------------
    inp = find_inputs(args.raw)
    required = ["train_features", "train_labels", "train_transcripts",
                "test_features", "test_labels", "test_transcripts"]
    missing = [k for k in required if not inp[k]]
    print("inputs:")
    for k in required + ["qualitative_metadata"]:
        names = ", ".join(str(p.relative_to(args.raw)) for p in inp[k]) or "NOT FOUND"
        print(f"  {k:<21} {names}{'' if k in required else '   (optional)'}")

    # ---- model files: vocabularies first (their bigrams are looked up in the transcripts) ---
    files = []
    skipped = defaultdict(int)                   # what was not entered (folders) or skipped (files)
    raw_abs = args.raw.resolve()                 # the --raw folder is not entered if it is inside
    raw_inside = raw_abs != repo and repo in raw_abs.parents and raw_abs != repo / "data/raw"
    for root, dirs, names in os.walk(repo):
        rel_root = Path(root).relative_to(repo).as_posix()
        rel_root = "" if rel_root == "." else rel_root + "/"
        keep = []
        for d in sorted(dirs):
            rel = rel_root + d + "/"
            if d == "__pycache__":
                skipped["__pycache__/"] += 1
            elif not rel_root and d.startswith(".venv"):
                skipped[".venv*/"] += 1
            elif rel_root == "data/raw/":
                skipped["data/raw/ subfolders"] += 1
            elif raw_inside and (raw_abs == (Path(root) / d).resolve()
                                 or raw_abs in (Path(root) / d).resolve().parents):
                skipped[rel + " (the --raw folder)"] += 1
            elif rel in SKIP_PREFIXES and rel != "data/raw/":
                skipped[rel] += 1
            else:
                keep.append(d)
        dirs[:] = keep
        for f in sorted(names):
            rel = rel_root + f
            if rel_root == "data/raw/" and rel not in SKIP_KEEP:
                skipped["data/raw/ (apart from README.md)"] += 1
                continue
            files.append((rel, Path(root) / f))
    files.sort()

    joblibs = {}
    by_sha: dict[str, str] = {}
    vocab_rows = []
    import joblib as _joblib
    for rel, p in files:
        if not rel.endswith(".joblib"):
            continue
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if sha in by_sha:
            joblibs[rel] = ("same", by_sha[sha]); continue
        by_sha[sha] = rel
        try:
            obj = _joblib.load(p)
        except Exception as e:                                         # noqa: BLE001
            joblibs[rel] = ("error", f"could not load ({type(e).__name__}: {e})"); continue
        vecs, arrays = walk_joblib(obj)
        joblibs[rel] = ("ok", arrays)
        for path, v in vecs:
            vocab_rows.append((rel, path, v.analyzer, v.ngram_range, vocab_stats(v, hasher)))
        del obj
    all_bigrams = (np.unique(np.concatenate([r[4]["bigram_hashes"] for r in vocab_rows]))
                   if vocab_rows else np.empty(0, np.uint64))

    # ---- index ----------------------------------------------------------------------------
    sources = inp["train_transcripts"] + inp["test_transcripts"]
    long_idx, short_idx, st, seen_bigram = build_index(sources, hasher, all_bigrams)
    long_los, short_los, prefix = load_objectives(inp["train_features"] + inp["test_features"], hasher)
    labels, orders = {}, {}
    for kind in ("train", "test"):
        d, o = load_labels(inp[f"{kind}_features"], inp[f"{kind}_labels"])
        if d:
            labels[kind], orders[kind] = d, o
    ids: set[str] = set()
    for f in inp["qualitative_metadata"]:
        q = pd.read_csv(f)
        for c in ("student_id", "tutor_id", "question_id"):
            if c in q.columns:
                ids |= {s for s in q[c].dropna().astype(str) if len(s) >= 6}
    u = st.get("utterances", 0)
    print("index:")
    print(f"  {u:,} utterances: {st.get('tokens >= 12', 0):,} of 12+ tokens, "
          f"{st.get('tokens 8-11', 0):,} of 8-11 tokens, {st.get('tokens < 8', 0):,} of fewer than 8 "
          f"tokens (not indexed)")
    print(f"  {len(long_idx):,} distinct {LONG_N}-token runs (>= {LONG_MIN_WORDS} words); "
          f"{len(short_idx):,} of {st.get('short utterances (distinct)', 0):,} distinct utterances of "
          f"{SHORT_N}-{LONG_N - 1} tokens (>= {SHORT_MIN_WORDS} words) occur at most {SHORT_MAX_COUNT} "
          f"times and are indexed whole")
    print(f"  {len(long_los) + len(short_los)} distinct objective texts (as token sequences): "
          f"{len(long_los)} of {LO_MIN_ANYWHERE}+ "
          f"tokens (matched anywhere), {len(short_los)} of 2-3 tokens (whole line or cell only)")
    print("  labels: " + (", ".join(f"{len(v):,} {k}" for k, v in labels.items()) or "none")
          + f"; {len(ids):,} qualitative_metadata ids")

    # ---- public wording: runs that also occur in pinned third-party texts leave the index ------
    ref_flags: dict[str, list[str]] = {}
    n_long, n_short = len(long_idx), len(short_idx)
    used, public_los = [], set()
    for rel, want in PUBLIC_REFERENCES.items():
        p = repo / rel
        if not p.is_file():
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != want:
            ref_flags[rel] = ["public reference file differs from its pinned sha256 (not used as reference)"]
            continue
        toks = TOK.findall(p.read_text(errors="ignore").lower())
        h, w = hasher.encode(toks)
        z = np.zeros(1, np.int64)
        long_idx = long_idx[~member(np.unique(ngrams(h, w, z, LONG_N, LONG_MIN_WORDS)[0]), long_idx)]
        short_idx = short_idx[~member(np.unique(short_runs(h, w, z)[0]), short_idx)]
        for lo, _ in lo_matches(toks, h, w, prefix):          # objective texts that are public wording
            public_los.add(lo)
        lines = {tuple(TOK.findall(x.lower())) for x in p.read_text(errors="ignore").splitlines()}
        lines |= {tuple(TOK.findall(c.lower())) for row in csv.reader(io.StringIO(p.read_text(errors="ignore")))
                  for c in row} if rel.endswith(".csv") else set()
        public_los |= set(short_los) & lines
        used.append(rel)
    print(f"  public wording: {n_long - len(long_idx):,} {LONG_N}-token runs and {n_short - len(short_idx):,} "
          f"short utterances also occur in {len(used)} pinned third-party file(s) and are not counted "
          f"as competition text: " + (", ".join(used) or "none"))
    if public_los:
        long_los = {k: v for k, v in long_los.items() if k not in public_los}
        short_los = {k: v for k, v in short_los.items() if k not in public_los}
        prefix = {hv: [k for k in ks if k not in public_los] for hv, ks in prefix.items()}
        prefix = {hv: ks for hv, ks in prefix.items() if ks}
    print(f"  {len(public_los)} objective text(s) occur verbatim in those files (curriculum wording) and are "
          f"not counted: {len(long_los) + len(short_los)} remain")

    sc = Scanner(dict(hasher=hasher, long_idx=long_idx, short_idx=short_idx, long_los=long_los,
                      short_los=short_los, lo_prefix=prefix, labels=labels, orders=orders, ids=ids))

    # ---- scan -----------------------------------------------------------------------------
    hits: dict[str, list[str]] = defaultdict(list)
    shows: dict[str, list] = defaultdict(list)
    counts = defaultdict(int)
    not_scanned = defaultdict(list)
    tree = repo / "models/submission"
    for rel, p in files:
        h, show = hits[rel], shows[rel]
        name = p.name
        try:
            if name.endswith(".joblib"):
                kind, val = joblibs[rel]
                if kind == "same":
                    counts["joblib (same bytes as another, checked once)"] += 1
                elif kind == "error":
                    h.append(val)
                else:
                    for path, a in val:
                        sc.check_array(a, f"array {path}", h)
                    counts["joblib"] += 1
            elif name.endswith((".pkl", ".pkl.gz", ".parquet")):
                df = pd.read_parquet(p) if name.endswith(".parquet") else pd.read_pickle(p)
                if isinstance(df, pd.DataFrame):
                    for col, txt in table_texts(df):
                        sc.scan_text(txt, f"column {col!r}, ", h, show, lines=False)
                        sc_short_cells(sc, df[col], col, h)
                    sc.check_label_columns(df, h)
                    counts["table"] += 1
                else:
                    not_scanned["pickle that is not a DataFrame"].append(rel)
                del df
            elif name.endswith(".npy"):
                sc.check_array(np.load(p, allow_pickle=False), "array", h)
                counts["npy"] += 1
            elif name.endswith(".zip"):
                scan_zip(sc, p, tree, h, show, counts, not_scanned, rel)
            elif name.endswith(WEIGHT_EXT):
                not_scanned["model weights (not scanned)"].append(rel)
            else:
                data = p.read_bytes()
                if name.endswith(".gz") and data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
                if is_binary(data):
                    not_scanned["binary (not scanned)"].append(rel); continue
                text = data.decode("utf-8", errors="ignore")
                delim = name.endswith((".csv", ".tsv"))
                sc.scan_text(text, "", h, show, cells=delim)
                if delim:
                    check_delimited(sc, text, "\t" if name.endswith(".tsv") else ",", h)
                counts["text"] += 1
        except Exception as e:                                         # noqa: BLE001
            h.append(f"could not read ({type(e).__name__}: {e})")
        if args.show and show:
            text = p.read_text(errors="ignore").splitlines() if not name.endswith(
                (".pkl", ".pkl.gz", ".parquet", ".zip")) else []
            for ln, n in show:
                if ln - 1 < len(text):
                    print(f"[show] {rel}:{ln} ({n}-token run): {text[ln - 1][:300]}", file=sys.stderr)

    print(f"scanned {sum(counts.values())} files: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    print("not scanned:")
    for k, v in skipped.items():
        unit = "file(s)" if k == "data/raw/ (apart from README.md)" else "folder(s), not entered"
        print(f"  {k}: {v} {unit} (not part of the repository)")
    for k, v in not_scanned.items():
        print(f"  {k}: {len(v)} files: " + ", ".join(v[:10]) + (" ..." if len(v) > 10 else ""))

    # ---- whitelist --------------------------------------------------------------------------
    flagged = dict(ref_flags)
    wl_hits: dict[str, list[str]] = defaultdict(list)
    for rel, h in hits.items():
        if not h:
            continue
        wl = next((w for w in WHITELIST if rel.startswith(w)), None)
        if wl:
            wl_hits[wl].append(f"{rel}: " + "; ".join(h[:4]))
        else:
            flagged[rel] = flagged.get(rel, []) + h
    for w, reason in WHITELIST.items():
        print(f"whitelist {w}*: {reason}")
        print("  hits in this run:" + ("" if wl_hits[w] else " none"))
        for line in wl_hits[w]:
            print(f"    {line}")
        for f, want in BASE_TOKENIZER_SHA.items():
            a = repo / w / f
            got = hashlib.sha256(a.read_bytes()).hexdigest() if a.is_file() else None
            print(f"  {f}: " + ("identical to the base model's file" if got == want
                                else "MISSING" if got is None else "DIFFERS from the base model's file"))
            if got != want:
                flagged.setdefault(w + f, []).append("whitelisted tokenizer file is not the base model's")

    # ---- vocabulary report (informational) ----------------------------------------------------
    print("vocabulary report (informational, not part of the result):")
    pos = {int(x): i for i, x in enumerate(all_bigrams)}
    lo_bigrams = set()                           # all objective texts, tokenised as scikit-learn does
    for f in inp["train_features"] + inp["test_features"]:
        for t in pd.read_csv(f, usecols=["learning_objective"]).learning_objective.dropna().astype(str):
            w = WORD.findall(t.lower())
            lo_bigrams |= {f"{x} {y}" for x, y in zip(w, w[1:])}
    for rel, (kind, val) in joblibs.items():
        if kind == "same":
            print(f"  {rel}: same bytes as {val}")
    for rel, path, analyzer, ngr, s in vocab_rows:
        line = (f"  {rel} {path}: {analyzer} {ngr[0]}-{ngr[1]}, {s['n']:,} terms, "
                f"{'sorted' if s['sorted'] else 'NOT SORTED (first-occurrence order)'}")
        if s["bigrams"]:
            in_tr = int(sum(seen_bigram[pos[int(x)]] for x in s["bigram_hashes"]))
            in_lo = sum(b in lo_bigrams for b in s["bigrams"])
            line += (f"; {len(s['bigrams']):,} word bigrams, {in_tr:,} of them occur in transcripts, "
                     f"{in_lo:,} in objective texts; longest chain of consecutive keys: "
                     f"{s['chain_words']} words")
        print(line)
    n_unsorted = sum(not r[4]["sorted"] for r in vocab_rows)
    print(f"  {len(vocab_rows)} fitted vectorizers, {n_unsorted} not sorted")

    # ---- result -----------------------------------------------------------------------------
    if flagged:
        print("files to review:")
    for rel, h in flagged.items():
        print(f"  {rel}: " + "; ".join(h[:4]))
    scope = (f"no {LONG_N}-token run of a transcript utterance, no rare whole utterance of "
             f"{SHORT_N}-{LONG_N - 1} tokens, no objective text, "
             f"no label column or array and no qualitative_metadata id found outside the whitelist; "
             f"model weights not scanned")
    if missing:
        print(f"RESULT: INCOMPLETE — required input(s) not found: {', '.join(missing)}"
              + (f"; {len(flagged)} file(s) to review (above)" if flagged else ""))
        sys.exit(2)
    if flagged:
        print(f"RESULT: {len(flagged)} file(s) to review (above)")
        sys.exit(1)
    print(f"RESULT: PASSED (scope: {scope})")
    sys.exit(0)


def sc_short_cells(sc: Scanner, col: pd.Series, name: str, hits: list) -> None:
    """Short objective texts as whole cells of a table's text column."""
    if not sc.short_los or not (pd.api.types.is_string_dtype(col) or col.dtype == object):
        return
    found = {}
    for v in pd.unique(col.dropna().astype(str)):
        k = tuple(TOK.findall(v.lower()))
        if k in sc.short_los:
            found[sc.short_los[k]] = 1
    if found:
        hits.append(f"short objective text as a whole cell: {len(found)} objective(s) "
                    f"(column {name!r}, e.g. id {next(iter(found))})")


def check_delimited(sc: Scanner, text: str, sep: str, hits: list) -> None:
    """Label check for a CSV/TSV file with a response_id column or exactly as many rows as a split."""
    head = text.split("\n", 1)[0].strip().replace('"', "").split(sep)
    n_rows = text.count("\n") - (0 if text.endswith("\n") else -1) - 1
    sizes = {len(y) for o in sc.orders.values() for _, y in o}
    if "response_id" in head or n_rows in sizes:
        df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
        sc.check_label_columns(df, hits)


def scan_zip(sc, p, tree, h, show, counts, not_scanned, rel) -> None:
    """Scan the members of a zip that are not byte-identical to their file in ``tree``.

    Args:
        sc: the Scanner.
        p: path of the zip.
        tree: folder whose files the members are compared with (``models/submission/``).
        h: hit list of this zip.
        show: (line, run length) list for ``--show``.
        counts: scan counters, updated in place.
        not_scanned: {reason: [paths]}, updated in place with binary members that differ.
        rel: the zip's path relative to the repository, used in messages.
    """
    with zipfile.ZipFile(p) as zf:
        same = other = 0
        for m in zf.infolist():
            if m.is_dir():
                continue
            data = zf.read(m)
            twin = tree / m.filename
            if twin.is_file() and twin.stat().st_size == len(data) and twin.read_bytes() == data:
                same += 1
                continue
            other += 1
            if is_binary(data):
                not_scanned[f"binary zip member without an identical twin in {tree.name}/ (not scanned)"] \
                    .append(f"{rel}:{m.filename}")
                continue
            sc.scan_text(data.decode("utf-8", errors="ignore"), f"member {m.filename}, ", h, show,
                         cells=m.filename.endswith((".csv", ".tsv")))
        h_note = f"{same} members identical to models/submission/, {other} scanned or listed"
        counts[f"zip ({h_note})"] += 1


if __name__ == "__main__":
    main()
