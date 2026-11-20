"""Extract provider-invariant per-session features from all transcripts.

Design rule after learning the test set is NOT all Third Space Learning:
every feature must be a ratio, a rate, or a within-session-normalised
quantity that still means something on a different tutoring provider or
modality (e.g. typed chat). TSL-specific artifacts ([unclear] markers, the
'background' role, absolute timing conventions) are kept OUT of the
predictor and used only for a domain score that measures how TSL-like a
session is.

For each session, THREE feature rows are computed:
  normal    the transcript as-is
  no_time   timestamps stripped (simulates a provider without timing)
  no_bg     'background' turns removed and consecutive same-role turns
            merged (simulates a different segmentation convention)
The stressed variants exist only to test robustness; assets ship 'normal'.

Also saved: a hashed bag-of-words per session (2^15 dims) so style
clustering never needs the 1.2 GB corpus resident again, plus the domain
artifact features.

Output: Exp2/data/hardened.npz + hardened_hash.npz
"""

import csv
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy import sparse

from training_paths import DATA
from training_paths import TRANSCRIPTS as TDIR
WORD_RE = re.compile(r"[a-z']+")

UNCERTAIN = ("i don't know", "i dont know", "not sure", "confused",
             "don't understand", "dont understand", "don't get", "dont get")
AGREE = ("yes", "yeah", "yep", "ok", "okay", "mhm", "mm")

F_NAMES = [
    # structure, ratio/rate based
    "log_n_turns", "frac_student_turns", "student_char_share",
    "log_ratio_stu_tut_len", "log_med_student_words",
    "stu_short_frac", "stu_oneword_frac",
    "tutor_q_rate", "student_q_rate", "alternation_rate",
    "mean_tutor_run", "stu_digit_rate", "stu_share_late_minus_early",
    "log_total_stu_chars", "stu_uncertain_rate", "stu_agree_rate",
    "log_ratio_stu_tut_turns", "stu_digit_chars_per_word",
    # timing, validity gated (indices 17..24 imputed when invalid)
    "log_duration", "log_med_gap", "log_p90_gap", "log_gap_p90_over_med",
    "log_ratio_stu_tut_pregap", "frac_long_gap",
    "log_late_early_gap_ratio", "log_turns_per_min",
    "timing_valid",
]
TIMING_IDX = list(range(18, 26))          # everything except the flag

D_NAMES = ["unclear_rate", "background_frac", "timing_valid",
           "log_mean_turn_chars", "log_n_turns_abs"]


def toks(s):
    return WORD_RE.findall(s.lower())


def parse_ts(s):
    try:
        parts = [int(x) for x in str(s).strip().split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return None


def features(turns, use_time):
    """turns: list of (role, content, seconds_or_None). Returns 26-dim row."""
    n = len(turns)
    stu = [(c, t) for r, c, t in turns if r == "student"]
    tut = [(c, t) for r, c, t in turns if r != "student"]
    ns, nt = len(stu), max(len(tut), 1)
    s_chars = sum(len(c) for c, _ in stu)
    t_chars = sum(len(c) for c, _ in tut)
    s_words = [toks(c) for c, _ in stu]
    wlens = [len(w) for w in s_words]

    third = max(n // 3, 1)
    early = sum(len(c) for r, c, _ in turns[:third] if r == "student")
    late = sum(len(c) for r, c, _ in turns[-third:] if r == "student")
    e_tot = sum(len(c) for _, c, _ in turns[:third]) or 1
    l_tot = sum(len(c) for _, c, _ in turns[-third:]) or 1

    alt = sum(1 for i in range(1, n)
              if (turns[i][0] == "student") != (turns[i - 1][0] == "student"))
    runs, cur = [], 0
    for r, _, _ in turns:
        if r != "student":
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)

    low = " ".join(c for c, _ in stu).lower()
    unc = sum(low.count(k) for k in UNCERTAIN)
    agree = sum(1 for w in s_words
                if w and all(x in AGREE for x in w[:2]) and len(w) <= 2)
    digit_turns = sum(1 for c, _ in stu if any(ch.isdigit() for ch in c))

    row = [
        np.log1p(n),
        ns / max(n, 1),
        s_chars / max(s_chars + t_chars, 1),
        np.log((s_chars / max(ns, 1) + 1) / (t_chars / nt + 1)),
        np.log1p(np.median(wlens) if wlens else 0),
        (sum(1 for w in wlens if w < 4) / max(ns, 1)),
        (sum(1 for w in wlens if w == 1) / max(ns, 1)),
        sum(c.count("?") for c, _ in tut) / nt,
        sum(c.count("?") for c, _ in stu) / max(ns, 1),
        alt / max(n - 1, 1),
        float(np.mean(runs)) if runs else 0.0,
        digit_turns / max(ns, 1),
        late / l_tot - early / e_tot,
        np.log1p(s_chars),
        unc / max(ns, 1),
        agree / max(ns, 1),
        np.log((ns + 1) / (nt + 1)),
        # organizers' "productive math talk" intensity feature: digit chars
        # per student word (narrated calculation vs bare answers)
        sum(ch.isdigit() for c, _ in stu for ch in c) / max(sum(wlens), 1),
    ]

    ts = [t for _, _, t in turns if t is not None] if use_time else []
    valid = (len(ts) >= 0.8 * n and len(set(ts)) > 3
             and n > 3 and (max(ts) - min(ts)) > 0)
    if valid:
        gaps = np.maximum(np.diff([t for _, _, t in turns
                                   if t is not None]), 0).astype(float)
        med = float(np.median(gaps))
        p90 = float(np.percentile(gaps, 90))
        pre_s = [max(turns[i][2] - turns[i - 1][2], 0)
                 for i in range(1, n)
                 if turns[i][0] == "student"
                 and turns[i][2] is not None and turns[i - 1][2] is not None]
        pre_t = [max(turns[i][2] - turns[i - 1][2], 0)
                 for i in range(1, n)
                 if turns[i][0] != "student"
                 and turns[i][2] is not None and turns[i - 1][2] is not None]
        dur = max(ts) - min(ts)
        half = len(gaps) // 2 or 1
        row += [
            np.log1p(dur),
            np.log1p(med),
            np.log1p(p90),
            np.log((p90 + 1) / (med + 1)),
            np.log((np.median(pre_s) + 1) / (np.median(pre_t) + 1)
                   ) if pre_s and pre_t else 0.0,
            float(np.mean(gaps > 3 * max(med, 1))),
            np.log((np.mean(gaps[half:]) + 1) / (np.mean(gaps[:half]) + 1)),
            np.log1p(60.0 * n / max(dur, 1)),
            1.0,
        ]
    else:
        row += [np.nan] * 8 + [0.0]        # imputed to train median later
    return row


def domain_row(turns_raw, timing_valid):
    n = len(turns_raw)
    unclear = sum(c.lower().count("[unclear]") for _, c, _ in turns_raw)
    bg = sum(1 for r, _, _ in turns_raw if r == "background")
    chars = sum(len(c) for _, c, _ in turns_raw)
    return [unclear / max(n, 1), bg / max(n, 1), timing_valid,
            np.log1p(chars / max(n, 1)), np.log1p(n)]


def merge_no_bg(turns):
    out = []
    for r, c, t in turns:
        if r == "background":
            continue
        if out and out[-1][0] == r:
            out[-1] = (r, out[-1][1] + " " + c, out[-1][2])
        else:
            out.append((r, c, t))
    return out


def main():
    sessions = sorted(p.stem for p in TDIR.glob("*.csv"))
    print(f"{len(sessions)} transcripts", flush=True)
    t0 = time.time()
    Fn, Ft, Fb, Dm = [], [], [], []
    rows_h, cols_h, vals_h = [], [], []
    HB = 1 << 15
    for i, s in enumerate(sessions):
        with open(TDIR / f"{s}.csv", encoding="utf-8-sig", newline="") as f:
            turns = [(str(x.get("role", "")).strip().lower(),
                      x.get("content") or "",
                      parse_ts(x.get("timestamp")))
                     for x in csv.DictReader(f)]
        if not turns:
            turns = [("tutor", "", None)]
        Fn.append(features(turns, True))
        Ft.append(features(turns, False))
        Fb.append(features(merge_no_bg(turns), True))
        Dm.append(domain_row(turns, Fn[-1][-1]))
        cnt = {}
        for _, c, _ in turns:
            for w in toks(c):
                h = zlib.crc32(w.encode()) % HB   # stable across processes
                cnt[h] = cnt.get(h, 0) + 1
        for h, v in cnt.items():
            rows_h.append(i)
            cols_h.append(h)
            vals_h.append(np.log1p(v))
        if (i + 1) % 4000 == 0:
            print(f"  {i + 1}  [{time.time() - t0:.0f}s]", flush=True)

    X = sparse.csr_matrix(
        (vals_h, (rows_h, cols_h)), shape=(len(sessions), HB), dtype=np.float32)
    np.savez_compressed(
        DATA / "hardened.npz",
        sessions=np.array(sessions),
        Fn=np.array(Fn), Ft=np.array(Ft), Fb=np.array(Fb),
        Dm=np.array(Dm),
        f_names=np.array(F_NAMES), d_names=np.array(D_NAMES))
    sparse.save_npz(DATA / "hardened_hash.npz", X)
    print(f"done {time.time() - t0:.0f}s  Fn {np.array(Fn).shape}  "
          f"hash nnz {X.nnz}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
