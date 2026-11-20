"""Build the tutor-student dialogue dataset (dialogueKT structure).

Following Scarlatos et al.: a dialogue is a sequence of alternating tutor
turns t_j and student turns s_j; each (t_j, s_j) pair is one time step that
can later be annotated for correctness and KCs.

From each raw transcript:
  - drop 'background' turns (ambient noise/other speakers), but count them
  - merge consecutive same-role turns into one turn (keeps first timestamp)
  - emit pairs: tutor context turn -> student response turn, in order
  - a leading student turn (session opens with the student) pairs with an
    empty tutor turn, matching the paper's s_0 convention

Outputs (Exp2/data/):
  dialogs.jsonl  one row per session: {session_id, turns:[{role,text,t}],
                 n_background}
  pairs.jsonl    one row per pair: {session_id, pair_idx, n_pairs, tutor,
                 student, t_tutor, t_student, stu_words, stu_has_digits}
Prints corpus statistics at the end.
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from training_paths import DATA, TRANSCRIPTS
WORD_RE = re.compile(r"[a-z0-9']+", re.I)


def parse_ts(s):
    try:
        p = [int(x) for x in str(s).strip().split(":")]
    except ValueError:
        return None
    return sum(v * m for v, m in zip(p[::-1], (1, 60, 3600))) \
        if len(p) in (2, 3) else None


def main():
    labels = set()
    for r in csv.DictReader(open(DATA / "train_features_TMQTWsB.csv")):
        labels.add(r["session_id"])
    sessions = sorted(labels)
    print(f"{len(sessions)} sessions", flush=True)

    t0 = time.time()
    n_pairs_all, stu_words_all, digit_share = [], [], []
    with open(DATA / "dialogs.jsonl", "w") as fd, \
         open(DATA / "pairs.jsonl", "w") as fp:
        for k, s in enumerate(sessions):
            with open(TRANSCRIPTS / f"{s}.csv",
                      encoding="utf-8-sig", newline="") as f:
                raw = [(str(x.get("role", "")).strip().lower(),
                        (x.get("content") or "").strip(),
                        parse_ts(x.get("timestamp")))
                       for x in csv.DictReader(f)]
            n_bg = sum(1 for r, _, _ in raw if r == "background")
            turns = []
            for role, text, ts in raw:
                if role == "background" or not text:
                    continue
                if turns and turns[-1]["role"] == role:
                    turns[-1]["text"] += " " + text
                else:
                    turns.append({"role": role, "text": text, "t": ts})
            fd.write(json.dumps({"session_id": s, "turns": turns,
                                 "n_background": n_bg}) + "\n")

            pairs = []
            pending_tutor = None
            for tr in turns:
                if tr["role"] == "student":
                    pairs.append((pending_tutor, tr))
                    pending_tutor = None
                else:
                    pending_tutor = tr
            for i, (tt, st) in enumerate(pairs):
                w = len(WORD_RE.findall(st["text"]))
                fp.write(json.dumps({
                    "session_id": s, "pair_idx": i, "n_pairs": len(pairs),
                    "tutor": tt["text"] if tt else "",
                    "student": st["text"],
                    "t_tutor": tt["t"] if tt else None,
                    "t_student": st["t"],
                    "stu_words": w,
                    "stu_has_digits": any(c.isdigit() for c in st["text"]),
                }) + "\n")
                stu_words_all.append(w)
                digit_share.append(any(c.isdigit() for c in st["text"]))
            n_pairs_all.append(len(pairs))
            if (k + 1) % 4000 == 0:
                print(f"  {k + 1}  [{time.time() - t0:.0f}s]", flush=True)

    n = np.array(n_pairs_all)
    w = np.array(stu_words_all)
    print(f"\nsessions               {len(n)}")
    print(f"total pairs            {n.sum()}")
    print(f"pairs/session          mean {n.mean():.1f}  median {np.median(n):.0f}"
          f"  p10 {np.percentile(n, 10):.0f}  p90 {np.percentile(n, 90):.0f}")
    print(f"student words/pair     mean {w.mean():.1f}  median {np.median(w):.0f}")
    print(f"pairs with digits      {np.mean(digit_share):.3f}")
    print(f"1-word student turns   {np.mean(w == 1):.3f}")
    print(f"<=3-word student turns {np.mean(w <= 3):.3f}")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    sys.exit(main())
