#!/usr/bin/env python
r"""Stage 1: synthetic learning-objective bank (maths + physics), novelty-banded.

Generates UK-curriculum-style objectives with a local LLM (vLLM), embeds them with the BGE
encoder and bands each by its max cosine to the 398 real objectives (twin >= 0.95,
sibling 0.90-0.95, related 0.85-0.90, novel < 0.85); twins of real objectives are dropped.

    python scripts/gen_synth/make_objectives.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --data-dir data --bge inference/assets/bge-base-en-v1.5 --out outputs/synth/objectives.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MATH_STRANDS = [
    "place value and number sense", "addition and subtraction strategies",
    "multiplication and division", "fractions", "decimals and percentages",
    "ratio and proportion", "algebraic thinking and sequences",
    "geometry: properties of shapes", "geometry: position and direction",
    "measurement: length, mass, capacity", "measurement: time and money",
    "statistics: charts and tables", "probability basics", "negative numbers",
    "roman numerals and number systems", "estimation and rounding",
]
PHYS_STRANDS = [
    "forces and motion", "energy stores and transfers", "electricity: simple circuits",
    "light and shadows", "sound and vibration", "magnetism", "states of matter",
    "the Earth and space", "waves basics", "pressure and density",
    "heat and temperature", "simple machines and levers",
]

PROMPT = """You write learning objectives for UK school lessons (KS1-KS3), in the exact
style of these real examples:
{examples}

Write {n} distinct learning objectives for the {subject} strand: "{strand}".
Rules: one per line, no numbering, each 5-20 words, start with a verb or "To ",
concrete and assessable in a single 45-minute one-on-one lesson, varied year levels.
Output only the objectives."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bge", default="inference/assets/bge-base-en-v1.5")
    ap.add_argument("--n-math", type=int, default=300)
    ap.add_argument("--n-physics", type=int, default=120)
    ap.add_argument("--out", default="outputs/synth500/objectives.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    feat = pd.read_csv(Path(args.data_dir).expanduser() / "train_features.csv")
    real = sorted(feat["learning_objective"].str.strip().unique())
    print(f"[obj] {len(real)} real objectives loaded")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1500, seed=args.seed)

    jobs = []
    for subject, strands, total in (("maths", MATH_STRANDS, args.n_math),
                                    ("physics", PHYS_STRANDS, args.n_physics)):
        per = int(np.ceil(total / len(strands)))
        for strand in strands:
            ex = "\n".join(rng.choice(real, 8, replace=False))
            msgs = [{"role": "user", "content": PROMPT.format(
                examples=ex, n=per, subject=subject, strand=strand)}]
            try:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=True, enable_thinking=False)
            except (TypeError, ValueError):
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            jobs.append((subject, strand, ids))

    outs = llm.generate([{"prompt_token_ids": j[2]} for j in jobs], sampling_params=sp)
    rows = []
    for (subject, strand, _), o in zip(jobs, outs):
        for ln in o.outputs[0].text.splitlines():
            t = re.sub(r"^[\s\d.\-*•]+", "", ln).strip()
            if 4 <= len(t.split()) <= 22:
                rows.append({"subject": subject, "strand": strand, "learning_objective": t})
    df = pd.DataFrame(rows).drop_duplicates("learning_objective").reset_index(drop=True)
    print(f"[obj] {len(df)} unique objectives generated")

    # ---- novelty banding vs the real 398 (bge, same space as coverage/te)
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer(str(Path(args.bge).expanduser()))
    E_real = bge.encode(real, normalize_embeddings=True, show_progress_bar=False)
    E_syn = bge.encode(df["learning_objective"].tolist(), normalize_embeddings=True,
                       show_progress_bar=False)
    max_sim = (E_syn @ E_real.T).max(axis=1)
    df["max_sim_real"] = np.round(max_sim, 4)
    df["band"] = pd.cut(df["max_sim_real"], [-1, 0.85, 0.90, 0.95, 2],
                        labels=["novel", "related", "sibling", "twin"])
    df = df[df["band"] != "twin"].reset_index(drop=True)
    df["difficulty"] = np.round(rng.normal(0, 1, len(df)), 3)  # latent, fixed per objective
    df.insert(0, "learning_objective_id", [f"syn{i:04d}" for i in range(len(df))])

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[obj] band mix: {df['band'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"[obj] wrote {len(df)} -> {out}")


if __name__ == "__main__":
    main()
