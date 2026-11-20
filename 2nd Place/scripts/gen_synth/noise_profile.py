#!/usr/bin/env python
r"""Measure the ASR/diarisation noise and structure of the real transcripts.

Writes outputs/eda/noise_profile.json, which inject_noise.py uses so synthetic noise matches
measured rates instead of hand-tuned ones, plus the structural statistics (utterance lengths,
turn transitions, timestamps) the generator needs to size sessions.

    python scripts/gen_synth/noise_profile.py --transcripts-dir data/train_transcripts \
        --data-dir data --out outputs/eda/noise_profile.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TAG_RE = re.compile(r"\[speaker:\s*([^\]]+)\]", re.I)
BRACKET_RE = re.compile(r"\[([a-zA-Z][a-zA-Z ]{0,30})\]")
FILLERS = ["um", "uh", "hmm", "okay", "yeah", "like", "so"]
# word-boundary regexes: "Um," / "uh." must count (a plain .split() misses fillers with
# punctuation attached)
FILLER_RE = {w: re.compile(rf"\b{w}\b") for w in FILLERS}
UMUH_RE = re.compile(r"\b(?:um|uh)\b")
UMUH_START_RE = re.compile(r"^(?:um|uh)\b")
# backchannels ("Mm-hmm" -> tfidf "mm"/"hmm"/"mm hmm") and the ASR spelling "alright" are
# frequent in real transcripts and absent from generated ones
MMHMM_RE = re.compile(r"\bmm+-?hmm*\b")
ALRIGHT_RE = re.compile(r"\balright\b")
ALL_RIGHT_RE = re.compile(r"\ball right\b")
# Steering vocabulary through the prompt backfires (the model overuses any phrase it is
# shown, avoid-lists included), so inject_noise thins each phrase to the real per-session
# rate measured here. Lexicon: the generator's habitual phrases plus real tutors' praise words.
PRAISE_LEXICON = ["perfect", "well done", "very good", "good job", "brilliant",
                  "awesome", "amazing", "excellent", "lovely", "spot on",
                  "trickier", "exactly right", "one last", "one more",
                  "try another", "hang of", "makes sense", "what if",
                  "nailed it", "exactly", "solid", "absolutely", "thanks"]
PRAISE_RE = {p: re.compile(rf"\b{p}\b") for p in PRAISE_LEXICON}
# short tag-questions ending real tutor turns ("..., okay?") are the main way real tutors
# reach ~0.85 questions per turn; inject_noise appends them to close the gap
TAG_CANDIDATES = ("okay", "right", "yeah", "yes", "alright", "no", "see")


def q(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95, 0.99)):
    v = np.asarray(v, dtype=float)
    return {f"p{int(x*100)}": round(float(np.quantile(v, x)), 3) for x in qs} if len(v) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="outputs/eda/noise_profile.json")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.transcripts_dir).expanduser() / "*.csv")))
    n_utt_sess, dur_sess, words_role = [], [], {"tutor": [], "student": [], "background": []}
    trans = Counter()            # role -> role transitions
    role_counts = Counter()
    ts_deltas = []
    bracket_vocab = Counter()    # lowercase bracket annotations, excl speaker tags
    unclear_case = Counter()     # [unclear] vs [UNCLEAR]
    tag_rows_by_role = Counter() # csv-role of rows containing [Speaker:...]
    tags_per_row = []
    tag_names = Counter()        # tutor / background / <name>
    q_rows = Counter()           # role -> rows with '?'
    filler_counts = Counter()
    umuh_rows = Counter()        # role -> rows containing um/uh
    umuh_n, umuh_start = 0, 0    # occurrences; occurrences at row start
    mmhmm_rows = Counter()       # role -> rows containing a mm-hmm backchannel
    alright_n, all_right_n = 0, 0
    praise_n = Counter()         # phrase -> total occurrences
    tag_counts = Counter()       # tag word -> tutor rows ending "..., <tag>?"
    tutor_q_rows = 0             # tutor rows ending with "?"
    unclear_rate_sess = []       # per-session [unclear]-per-row (dispersion target)
    mojibake_rows = 0
    words_total = 0
    rows_total = 0
    bg_words = []

    for path in files:
        n = 0
        n_unclear_file = 0
        prev_role, first_ts, last_ts, prev_t = None, None, None, None
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                n += 1
                rows_total += 1
                role = (row.get("role") or "").strip().lower()
                c = row.get("content") or ""
                low = c.lower()
                nw = len(c.split())
                words_total += nw
                role_counts[role] += 1
                if role in words_role:
                    words_role[role].append(nw)
                if role == "background":
                    bg_words.append(nw)
                if prev_role is not None:
                    trans[f"{prev_role}->{role}"] += 1
                prev_role = role
                if "?" in c:
                    q_rows[role] += 1
                if "�" in c:
                    mojibake_rows += 1
                for w, rx in FILLER_RE.items():
                    filler_counts[w] += len(rx.findall(low))
                hits = UMUH_RE.findall(low)
                if hits and role in ("tutor", "student"):
                    umuh_rows[role] += 1
                    umuh_n += len(hits)
                    if UMUH_START_RE.match(low):
                        umuh_start += 1
                if MMHMM_RE.search(low) and role in ("tutor", "student"):
                    mmhmm_rows[role] += 1
                alright_n += len(ALRIGHT_RE.findall(low))
                all_right_n += len(ALL_RIGHT_RE.findall(low))
                for p, rx in PRAISE_RE.items():
                    praise_n[p] += len(rx.findall(low))
                if role == "tutor" and low.rstrip().endswith("?"):
                    tutor_q_rows += 1
                    lw = re.findall(r"[a-z']+", low)
                    if lw and lw[-1] in TAG_CANDIDATES:
                        tag_counts[lw[-1]] += 1
                tags = TAG_RE.findall(c)
                if tags:
                    tag_rows_by_role[role] += 1
                    tags_per_row.append(len(tags))
                    for t in tags:
                        t = t.strip().lower()
                        tag_names["tutor" if t == "tutor" else
                                  "background" if t == "background" else "name"] += 1
                for b in BRACKET_RE.findall(c):
                    bl = b.strip().lower()
                    if bl.startswith("speaker:"):
                        continue
                    if bl == "unclear":
                        unclear_case["UNCLEAR" if "[UNCLEAR]" in c else "unclear"] += 1
                        n_unclear_file += 1
                    bracket_vocab[bl] += 1
                try:
                    h, m, s = (row.get("timestamp") or "").strip().split(":")
                    t = int(h) * 3600 + int(m) * 60 + int(s)
                    if first_ts is None:
                        first_ts = t
                    if prev_t is not None and 0 <= t - prev_t <= 120:
                        ts_deltas.append(t - prev_t)
                    prev_t = t
                    last_ts = t
                except Exception:
                    pass
        n_utt_sess.append(n)
        if n:
            unclear_rate_sess.append(n_unclear_file / n)
        if first_ts is not None and last_ts is not None:
            dur_sess.append(last_ts - first_ts)

    # objectives-per-session mix (drives the latent sampler)
    import pandas as pd
    feat = pd.read_csv(Path(args.data_dir).expanduser() / "train_features.csv")
    n_resp = feat.groupby("session_id").size()
    lo_per_sess = {str(k): round(float(v), 4)
                   for k, v in (n_resp.value_counts(normalize=True).sort_index()
                                .head(6).items())}

    profile = {
        "sessions": len(files), "rows": rows_total, "words": words_total,
        "n_utt_per_session": q(n_utt_sess),
        "duration_s_per_session": q(dur_sess),
        "role_share": {k: round(v / rows_total, 4) for k, v in role_counts.items()},
        "words_per_utt": {k: q(v) for k, v in words_role.items()},
        "background_words": q(bg_words),
        "transitions_top": {k: round(v / max(rows_total - len(files), 1), 4)
                            for k, v in trans.most_common(12)},
        "ts_delta_s": q(ts_deltas),
        "question_row_rate": {k: round(q_rows[k] / role_counts[k], 4)
                              for k in ("tutor", "student") if role_counts[k]},
        "unclear_per_row": round(sum(unclear_case.values()) / rows_total, 5),
        "unclear_case_mix": {k: round(v / max(sum(unclear_case.values()), 1), 3)
                             for k, v in unclear_case.items()},
        "bracket_vocab_top": dict(bracket_vocab.most_common(15)),
        "speaker_tag_row_rate_by_role": {
            k: round(tag_rows_by_role[k] / role_counts[k], 5)
            for k in ("tutor", "student", "background") if role_counts[k]},
        "speaker_tags_per_tagged_row": q(tags_per_row, (0.5, 0.9, 0.99)),
        "speaker_tag_name_mix": {k: round(v / max(sum(tag_names.values()), 1), 3)
                                 for k, v in tag_names.items()},
        "mojibake_row_rate": round(mojibake_rows / rows_total, 5),
        "filler_per_1k_words": {k: round(v / words_total * 1000, 2)
                                for k, v in filler_counts.items()},
        "umuh_row_rate_by_role": {k: round(umuh_rows[k] / role_counts[k], 5)
                                  for k in ("tutor", "student") if role_counts[k]},
        "umuh_per_filler_row": round(umuh_n / max(sum(umuh_rows.values()), 1), 3),
        "umuh_start_share": round(umuh_start / max(sum(umuh_rows.values()), 1), 3),
        "mmhmm_rows_per_session_by_role": {
            k: round(mmhmm_rows[k] / len(files), 3) for k in ("tutor", "student")},
        "alright_spelling_share": round(alright_n / max(alright_n + all_right_n, 1), 3),
        "unclear_row_rate_per_session": q(unclear_rate_sess),
        "praise_per_session": {p: round(praise_n[p] / len(files), 3)
                               for p in PRAISE_LEXICON},
        "tutor_tag_mix": {t: round(v / max(sum(tag_counts.values()), 1), 3)
                          for t, v in tag_counts.most_common()},
        "tutor_tag_row_share": round(sum(tag_counts.values()) / max(tutor_q_rows, 1), 3),
        "objectives_per_session_mix": lo_per_sess,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
