#!/usr/bin/env python
r"""Stage 4: add measured ASR/diarisation noise and write the final CSVs and labels.

Every rate comes from outputs/eda/noise_profile.json (the real transcripts), scaled by the
session's tech_noise latent: [unclear] tokens, "Mm-hmm." backchannels, the "alright" spelling,
thinning of over-used phrases to real rates, tag-questions on tutor turns, background rows
(some misrouted real dialogue), diarisation merges behind [Speaker:X], silent role swaps on
short turns, and timestamps from the measured gap distribution, mildly rescaled to a sampled
session duration. Outputs: <out-dir>/transcripts/<sid>.csv (competition schema), features.csv,
labels.csv (is_correct + p_soft) and meta.jsonl.

    python scripts/gen_synth/inject_noise.py --clean outputs/synth/clean.jsonl \
        --sessions outputs/synth/sessions.jsonl --profile outputs/eda/noise_profile.json \
        --out-dir outputs/synth/pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

BG_SNIPPETS = ["[unclear]", "Hello.", "Yeah.", "Okay.", "[no speech]",
               "You need to finish your homework after.", "Dinner's ready soon.",
               "[music]", "Sorry, one second.", "That's the doorbell."]


def qsample(rng, qd, n=1):
    ps = sorted((int(k[1:]) / 100, v) for k, v in qd.items())
    u = rng.uniform(ps[0][0], ps[-1][0], n)
    return np.interp(u, [p for p, _ in ps], [v for _, v in ps])


def add_fillers(rng, rows, prof, mult):
    """Insert um/uh at the measured per-role row rates.

    The generator is told not to write them, so the rate is controlled here; `mult` is a
    per-session propensity (real speakers differ a lot in how much they um).
    """
    rates = prof.get("umuh_row_rate_by_role", {"tutor": 0.02, "student": 0.02})
    start_share = prof.get("umuh_start_share", 0.55)
    per_row = max(prof.get("umuh_per_filler_row", 1.1), 1.0)
    out = []
    for role, text in rows:
        p = rates.get(role, 0.0) * mult
        if p and rng.uniform() < min(p, 0.5):
            for _ in range(max(1, rng.poisson(per_row - 1) + 1)):
                w = "um" if rng.uniform() < 0.47 else "uh"  # measured mix
                words = text.split()
                if rng.uniform() < start_share or not words:
                    text = f"{w.capitalize()}, {text}"
                else:
                    pos = int(rng.integers(1, len(words) + 1))
                    words.insert(pos, f"{w},")
                    text = " ".join(words)
        out.append([role, text])
    return out


def add_backchannels(rng, rows, prof, mult):
    """Insert standalone "Mm-hmm." rows after the other role's turns, at measured rates.

    The tutor backchannels while the student works and vice versa. Like um/uh these are speech
    habits the generator is told not to write, so the rate is controlled here.
    """
    per_sess = prof.get("mmhmm_rows_per_session_by_role", {})
    variants = ["Mm-hmm.", "Mm-hmm.", "Mm-hmm, yeah.", "Mm."]
    ins = []  # (position, role)
    for role, other in (("tutor", "student"), ("student", "tutor")):
        lam = per_sess.get(role, 0.0) * mult
        n = min(rng.poisson(lam), 12) if lam else 0
        spots = [i for i, (r, t) in enumerate(rows)
                 if r == other and len(t.split()) >= 6]
        if n and spots:
            for i in rng.choice(spots, size=min(n, len(spots)), replace=False):
                ins.append((int(i) + 1, role))
    for pos, role in sorted(ins, reverse=True):
        rows.insert(pos, [role, str(rng.choice(variants))])
    return rows


def respell_alright(rng, text, share):
    return re.sub(r"\b([Aa])ll right\b",
                  lambda m: (("A" if m.group(1) == "A" else "a") + "lright")
                  if rng.uniform() < share else m.group(0), text)


def _strip_match(text, m):
    """Remove a phrase occurrence and stitch the sentence back together."""
    before, after = text[:m.start()].rstrip(), text[m.end():]
    after = re.sub(r"^[\s,.!:;-]+", "", after)
    if after and (not before or before.endswith((".", "!", "?"))):
        after = after[0].upper() + after[1:]
    return (before + " " + after).strip() if after else before


def thin_lexicon(rng, rows, prof, n_utt_p50):
    """Thin each over-used phrase to its measured real per-session rate (scaled by length).

    Steering vocabulary through the prompt fails in both directions (the model overuses any phrase
    it is shown), so rates are matched here. Surplus occurrences are deleted, never swapped
    (replacements just create new tells); a row left empty is dropped.
    """
    targets = prof.get("praise_per_session", {})
    if not targets:
        return rows
    scale = len(rows) / max(n_utt_p50, 1.0)
    # longest phrase first so deleting "exactly right" doesn't corrupt "exactly"
    for phrase in sorted(targets, key=len, reverse=True):
        rx = re.compile(rf"\b{re.escape(phrase)}\b", re.I)
        occ = [(i, m) for i, (_, tx) in enumerate(rows) for m in rx.finditer(tx)]
        surplus = len(occ) - rng.poisson(targets[phrase] * scale)
        if surplus <= 0:
            continue
        drop = rng.choice(len(occ), size=surplus, replace=False)
        for k in sorted(drop, key=lambda k: -occ[k][1].start()):  # right-to-left
            i, m = occ[k]
            # spans shift after edits within the same row: re-find to stay safe
            m2 = rx.search(rows[i][1])
            if m2:
                rows[i][1] = _strip_match(rows[i][1], m2)
    return [r for r in rows if r[1].strip()]


def add_tag_questions(rng, rows, prof):
    """Append short tag-questions (", okay?") to tutor turns up to the real question-row rate.

    The real tutor rate is ~0.54; generated text sits lower and prompt instructions do not move it.
    """
    mix = prof.get("tutor_tag_mix")
    target = prof.get("question_row_rate", {}).get("tutor")
    if not mix or not target:
        return rows
    tags, probs = zip(*mix.items())
    probs = np.asarray(probs) / sum(probs)
    tut = [i for i, (r, _) in enumerate(rows) if r == "tutor"]
    have = sum("?" in rows[i][1] for i in tut)
    cand = [i for i in tut if "?" not in rows[i][1]
            and len(rows[i][1].split()) >= 4 and not rows[i][1].endswith("?")]
    need = min(int(rng.poisson(max(target * len(tut) - have, 0))), len(cand))
    for i in rng.choice(cand, size=need, replace=False):
        tag = str(rng.choice(tags, p=probs))
        rows[i][1] = rows[i][1].rstrip().rstrip(".!,;") + f", {tag}?"
    return rows


def add_unclear(rng, text, rate, case_mix):
    """Insert/replace with [unclear] at ~`rate` per row (word-position uniform)."""
    words = text.split()
    n_ins = rng.poisson(rate)
    for _ in range(min(n_ins, 3)):
        tokv = "[unclear]" if rng.uniform() < case_mix.get("unclear", 0.87) else "[UNCLEAR]"
        pos = int(rng.integers(0, len(words) + 1))
        if words and rng.uniform() < 0.5:  # replace a word (ASR dropout)
            words[min(pos, len(words) - 1)] = tokv
        else:
            words.insert(pos, tokv)
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="outputs/synth500/clean.jsonl")
    ap.add_argument("--sessions", default="outputs/synth500/sessions.jsonl")
    ap.add_argument("--profile", default="outputs/eda/noise_profile.json")
    ap.add_argument("--out-dir", default="outputs/synth500/pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    prof = json.loads(Path(args.profile).read_text())

    sess_meta = {json.loads(ln)["session_id"]: json.loads(ln)
                 for ln in open(args.sessions, encoding="utf-8")}
    out_dir = Path(args.out_dir).expanduser()
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    tag_rate = float(np.mean(list(prof["speaker_tag_row_rate_by_role"].values())))
    name_mix = prof["speaker_tag_name_mix"]
    bg_share = prof["role_share"].get("background", 0.04)
    feats, labels, metas = [], [], []
    n_done = 0

    for ln in open(args.clean, encoding="utf-8"):
        rec = json.loads(ln)
        s = sess_meta[rec["session_id"]]
        k = s["tech_noise"]  # scales all noise ops for this session
        student = s["student"]
        rows = [[role, text] for role, text in rec["lines"]]

        # --- -1. register repair on the clean text: thin over-used phrases to real rates
        # and close the tutor question-rate gap with tag-questions
        rows = thin_lexicon(rng, rows, prof,
                            prof.get("n_utt_per_session", {}).get("p50", 267))
        rows = add_tag_questions(rng, rows, prof)

        # --- 0. speech habits (not tech noise -> own per-session multipliers):
        # um/uh fillers + mm-hmm backchannel rows + the "alright" ASR spelling
        rows = add_fillers(rng, rows, prof,
                           mult=float(np.clip(rng.lognormal(0, 0.4), 0.3, 3.0)))
        rows = add_backchannels(rng, rows, prof,
                                mult=float(np.clip(rng.lognormal(0, 0.5), 0.2, 4.0)))
        ar_share = prof.get("alright_spelling_share", 0.27)
        rows = [[ro, respell_alright(rng, tx, ar_share)] for ro, tx in rows]

        # --- 1. diarization merges: fold next turn into this row behind [Speaker:X]
        merged, i = [], 0
        while i < len(rows):
            if i + 1 < len(rows) and rng.uniform() < tag_rate * k:
                nrole, ntext = rows[i + 1]
                u = rng.uniform()
                tag = ("Tutor" if nrole == "tutor" else student) \
                    if u < name_mix.get("tutor", .45) + name_mix.get("name", .5) \
                    else "Background"
                merged.append([rows[i][0],
                               f"{rows[i][1]} [Speaker:{tag}] {ntext}"])
                i += 2
            else:
                merged.append(rows[i])
                i += 1
        rows = merged

        # --- 2. silent role swaps on short turns (the untagged error class)
        for r in rows:
            if len(r[1].split()) <= 12 and rng.uniform() < 0.008 * k:
                r[0] = "tutor" if r[0] == "student" else "student"

        # --- 3. background rows: inserted snippets + misrouted real dialogue
        with_bg = []
        for r in rows:
            if rng.uniform() < bg_share * 0.7 * k:
                with_bg.append(["background", str(rng.choice(BG_SNIPPETS))])
            if rng.uniform() < bg_share * 0.3 * k:  # real dialogue -> background
                r = ["background", r[1]]
            with_bg.append(r)
        rows = with_bg

        # --- 4. [unclear] noise on every row; the per-session rate is sampled from the
        # measured per-session distribution (IQR ~0.26-0.35 per row): a global rate times
        # tech_noise gives about 3x too much spread
        if "unclear_row_rate_per_session" in prof:
            rate = float(qsample(rng, prof["unclear_row_rate_per_session"])[0])
        else:
            rate = prof["unclear_per_row"] * k
        rows = [[ro, add_unclear(rng, tx, rate, prof["unclear_case_mix"])]
                for ro, tx in rows]

        # --- 5. timestamps: sample the measured inter-utterance gaps, then rescale mildly
        # toward a duration drawn from the real per-session distribution. The scale is
        # clamped to 0.7-1.3 so single gaps stay plausible; unscaled sums have too long a
        # tail, and a hard rescale to the target gives absurd gaps on short transcripts.
        deltas = qsample(rng, prof["ts_delta_s"], len(rows))
        target = float(qsample(rng, prof["duration_s_per_session"])[0])
        deltas = deltas * float(np.clip(target / max(deltas.sum(), 1.0), 0.7, 1.3))
        t, ts = 0.0, []
        for d in deltas:
            ts.append(int(t))
            t += d

        sid = rec["session_id"]
        with open(out_dir / "transcripts" / f"{sid}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["session_id", "utterance_id", "role", "content", "timestamp"])
            for j, ((ro, tx), tj) in enumerate(zip(rows, ts)):
                w.writerow([sid, j, ro, tx,
                            f"{tj//3600:02d}:{tj%3600//60:02d}:{tj%60:02d}"])
        for o in s["objectives"]:
            rid = f"r{sid}{o['learning_objective_id']}"
            feats.append({"response_id": rid, "session_id": sid,
                          "learning_objective_id": o["learning_objective_id"],
                          "learning_objective": o["learning_objective"]})
            labels.append({"response_id": rid, "is_correct": o["is_correct"],
                           "p_soft": o["p_correct"]})
        metas.append({"session_id": sid, **{k_: s[k_] for k_ in
                      ("ability", "engagement", "tech_noise")},
                      "exemplar_src": rec.get("exemplar_src"),
                      "bands": [o["band"] for o in s["objectives"]],
                      "subjects": [o["subject"] for o in s["objectives"]]})
        n_done += 1

    import pandas as pd
    pd.DataFrame(feats).to_csv(out_dir / "features.csv", index=False)
    pd.DataFrame(labels).to_csv(out_dir / "labels.csv", index=False)
    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    print(f"[noise] {n_done} transcripts + features/labels/meta -> {out_dir}")


if __name__ == "__main__":
    main()