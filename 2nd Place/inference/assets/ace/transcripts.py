"""Shared transcript parsing — single implementation for train (EDA) and inference.

The raw statistics reproduce exactly the definitions used to build
outputs/eda/transcript_stats.csv (run_eda.py), which the GBM was trained on:
word counts use whitespace split of the lowercased content, `w_total` includes
`background` rows, the token set spans ALL roles, `[unclear]`/bracket counts are
per-occurrence on the lowercased text, and duration is last-first parseable timestamp.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

BRACKET_RE = re.compile(r"\[([a-z][a-z ]{0,30})\]")
TOKEN_RE = re.compile(r"[a-z']+")
STOPWORDS = set("""a an and are as at be been being by can could do does did for from
has have had he her him his how i if in into is it its me my not of on or our so
she that the their them then there they this to up us was we were what when which
who will with would you your yours it's don't i'm that's""".split())

SEG_WORDS = 200
SEG_STRIDE = 150
SEG_MIN_WORDS = 30

RAW_STAT_KEYS = ["n_utt", "n_tutor", "n_student", "n_background", "w_total",
                 "w_student", "w_tutor", "dur_s", "tutor_q", "student_q",
                 "n_unclear", "n_bracket"]


def _parse_ts(ts: str):
    try:
        h, m, s = ts.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None


def parse_transcript(path_str: str):
    """One pass over a transcript CSV.

    Returns (session_id, stats_dict, ts_text, token_set):
      stats_dict — RAW_STAT_KEYS, matching run_eda definitions
      ts_text    — tutor+student utterances joined by " " (TF-IDF / segmentation input)
      token_set  — lowercase [a-z'] tokens over ALL roles (lexical-overlap input)
    """
    path = Path(path_str)
    sid = path.stem
    st = {k: 0 for k in RAW_STAT_KEYS}
    ts_parts: list[str] = []
    tok_set: set[str] = set()
    first_ts = last_ts = None
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                role = (row.get("role") or "").strip()
                content = row.get("content") or ""
                st["n_utt"] += 1
                low = content.lower()
                nw = len(low.split())
                st["w_total"] += nw
                st["n_unclear"] += low.count("[unclear]")
                st["n_bracket"] += len(BRACKET_RE.findall(low))
                tok_set.update(TOKEN_RE.findall(low))
                if role == "tutor":
                    st["n_tutor"] += 1
                    st["w_tutor"] += nw
                    st["tutor_q"] += content.count("?")
                    ts_parts.append(content)
                elif role == "student":
                    st["n_student"] += 1
                    st["w_student"] += nw
                    st["student_q"] += content.count("?")
                    ts_parts.append(content)
                elif role == "background":
                    st["n_background"] += 1
                t = _parse_ts(row.get("timestamp") or "")
                if t is not None:
                    if first_ts is None:
                        first_ts = t
                    last_ts = t
    except FileNotFoundError:
        pass
    st["dur_s"] = (last_ts - first_ts) if first_ts is not None else np.nan
    return sid, st, " ".join(ts_parts), tok_set


def segment_text(ts_text: str,
                 seg_words: int = SEG_WORDS,
                 stride: int = SEG_STRIDE,
                 min_words: int = SEG_MIN_WORDS) -> list[str]:
    """200-word windows, stride 150 — identical to the embedding pass (run_embeddings)."""
    w = ts_text.split()
    if not w:
        return []
    segs = []
    for i in range(0, max(1, len(w) - stride), stride):
        seg = w[i:i + seg_words]
        if len(seg) >= min_words or i == 0:
            segs.append(" ".join(seg))
    return segs


def objective_words(lo_text: str) -> frozenset:
    """Content words of a learning objective (for the lexical-overlap feature)."""
    return frozenset(w for w in TOKEN_RE.findall(str(lo_text).lower())
                     if w not in STOPWORDS and len(w) > 2)


def lexical_overlap(lo_text: str, token_set: set[str]) -> float:
    words = objective_words(lo_text)
    if not words:
        return np.nan
    return sum(1 for w in words if w in token_set) / len(words)