#!/usr/bin/env python
r"""Stage 2: sample session latents and labels before any transcript is written.

Per session: student and tutor personas, ability, engagement, tech-noise level, target length
and 1-3 objectives (from the measured profile). Per (session, objective): p_correct from a noisy
IRT-style model calibrated to the train prior (0.7025), a hard label ~ Bernoulli(p), and an
evidence brief for the generator. p is kept as a soft label.

    python scripts/gen_synth/sample_latents.py --objectives outputs/synth/objectives.csv \
        --profile outputs/eda/noise_profile.json --n 300 --out outputs/synth/sessions.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDENTS = ["Alfie", "Amara", "Bea", "Callie", "Dexter", "Elias", "Femi", "Grace",
            "Hassan", "Imogen", "Jude", "Kwame", "Lena", "Marcus", "Nadia", "Oscar",
            "Priya", "Quinn", "Rory", "Saanvi", "Teo", "Uma", "Viktor", "Willow",
            "Xavier", "Yusuf", "Zara", "Idris", "Maeve", "Otis"]
TUTOR_STYLES = [
    "warm and encouraging, uses lots of praise and check-in questions",
    "brisk and structured, moves quickly through activities",
    "patient and Socratic, answers questions with questions",
    "chatty and informal, uses everyday examples",
    "methodical, insists on the student explaining their reasoning",
]
EVIDENCE = [  # (p threshold, brief for the generator)
    (0.85, "confident mastery: answers mostly correct and quick, explains reasoning, "
           "at most one small slip that they self-correct"),
    (0.65, "solid but imperfect: gets most questions right, a couple of errors fixed "
           "after a hint, explanation sometimes shaky"),
    (0.45, "genuinely mixed and ambiguous: some correct answers, some errors, needs "
           "several hints, hard to tell if the idea has stuck by the end"),
    (0.25, "struggling: frequent errors and long pauses, tutor re-teaches and "
           "simplifies, only partial success on easier variants"),
    (-1.0, "clear difficulty: mostly incorrect even after re-teaching, guesses, "
           "session ends with the objective not secured"),
]


def qsample(rng, qd, n=1):
    """Sample via piecewise-linear inverse CDF from a {p5:..,p25:..,...} dict."""
    ps = sorted((int(k[1:]) / 100, v) for k, v in qd.items())
    u = rng.uniform(ps[0][0], ps[-1][0], n)
    return np.interp(u, [p for p, _ in ps], [v for _, v in ps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", default="outputs/synth500/objectives.csv")
    ap.add_argument("--profile", default="outputs/eda/noise_profile.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--target-prior", type=float, default=0.7025)
    ap.add_argument("--subject-mix", default="maths=0.9,physics=0.1",
                    help="session subject probabilities (real train is all maths; "
                         "physics exists for the novel-skill stress pool)")
    ap.add_argument("--band-mix", default="related=0.725,novel=0.225,sibling=0.05",
                    help="objective novelty-band weights within a session's subject "
                         "pool; the default puts the overall mix near the test-like one "
                         "(about 68%% related, 32%% novel) once the all-novel physics share "
                         "dilutes it. 'uniform' disables the weighting.")
    ap.add_argument("--out", default="outputs/synth500/sessions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    bank = pd.read_csv(args.objectives)
    prof = json.loads(Path(args.profile).read_text())

    subj_mix = {k: float(v) for k, v in
                (kv.split("=") for kv in args.subject_mix.split(","))}
    subj_names = [s for s in subj_mix if (bank["subject"] == s).any()]
    subj_p = np.array([subj_mix[s] for s in subj_names])
    subj_p = subj_p / subj_p.sum()
    if args.band_mix == "uniform":
        bank["w"] = 1.0
    else:
        band_mix = {k: float(v) for k, v in
                    (kv.split("=") for kv in args.band_mix.split(","))}
        # weight so each band's total draw probability matches the mix, within subject
        bank["w"] = 0.0
        for subj, g in bank.groupby("subject"):
            for band, gg in g.groupby("band", observed=True):
                bank.loc[gg.index, "w"] = band_mix.get(str(band), 0.0) / len(gg)
    lo_mix = {int(k): v for k, v in prof["objectives_per_session_mix"].items() if int(k) <= 3}
    ks, ws = list(lo_mix), np.array(list(lo_mix.values()))
    ws = ws / ws.sum()

    # calibrate the IRT intercept so mean(p) hits the train prior
    def mean_p(b0, n=20000):
        r = np.random.default_rng(0)
        z = (b0 + 1.1 * r.normal(0, 1, n) - 0.9 * r.normal(0, 1, n)
             + 0.3 * r.normal(0, 1, n) + r.normal(0, 0.5, n))
        return float((1 / (1 + np.exp(-z))).mean())
    lo_b, hi_b = -2.0, 4.0
    for _ in range(40):
        mid = (lo_b + hi_b) / 2
        lo_b, hi_b = (mid, hi_b) if mean_p(mid) < args.target_prior else (lo_b, mid)
    b0 = (lo_b + hi_b) / 2
    print(f"[latents] intercept {b0:.3f} -> mean p {mean_p(b0):.4f}")

    seen_sids: set[str] = set()
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            ability = rng.normal()
            engagement = rng.normal()
            n_lo = int(rng.choice(ks, p=ws))
            # sessions mostly stay in one subject, like real curricula
            subj = str(rng.choice(subj_names, p=subj_p))
            pool = bank[(bank["subject"] == subj) & (bank["w"] > 0)]
            chosen = pool.sample(min(n_lo, len(pool)), weights=pool["w"],
                                 random_state=int(rng.integers(1e9)))
            objectives = []
            for _, o in chosen.iterrows():
                z = (b0 + 1.1 * ability - 0.9 * o["difficulty"]
                     + 0.3 * engagement + rng.normal(0, 0.5))
                p = float(1 / (1 + np.exp(-z)))
                objectives.append({
                    "learning_objective_id": o["learning_objective_id"],
                    "learning_objective": o["learning_objective"],
                    "subject": o["subject"], "band": str(o["band"]),
                    "p_correct": round(p, 4),
                    "is_correct": int(rng.uniform() < p),
                    "evidence": next(br for th, br in EVIDENCE if p >= th),
                })
            # birthday collisions happen at this scale (4 duplicates at n=8000) and later
            # stages key on session_id, so re-draw until unique
            sid = "zz" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 5))
            while sid in seen_sids:
                sid = "zz" + "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 5))
            seen_sids.add(sid)
            sess = {
                "session_id": sid,
                "student": str(rng.choice(STUDENTS)),
                "tutor_style": str(rng.choice(TUTOR_STYLES)),
                "ability": round(float(ability), 3),
                "engagement": round(float(engagement), 3),
                "tech_noise": round(float(np.clip(rng.gamma(2, 0.5), 0.2, 3.0)), 2),
                "n_utt_target": int(qsample(rng, prof["n_utt_per_session"])[0]),
                "duration_target_s": int(qsample(rng, prof["duration_s_per_session"])[0]),
                "objectives": objectives,
            }
            f.write(json.dumps(sess) + "\n")
    print(f"[latents] wrote {args.n} sessions -> {args.out}")
    rows = [o for ln in open(args.out, encoding="utf-8")
            for o in json.loads(ln)["objectives"]]
    print(f"[latents] realized prior {np.mean([o['is_correct'] for o in rows]):.4f} "
          f"over {len(rows)} rows")
    print(f"[latents] band mix: "
          f"{pd.Series([o['band'] for o in rows]).value_counts(normalize=True).round(3).to_dict()}")
    print(f"[latents] subject mix: "
          f"{pd.Series([o['subject'] for o in rows]).value_counts(normalize=True).round(3).to_dict()}")


if __name__ == "__main__":
    main()