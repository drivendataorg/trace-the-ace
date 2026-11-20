"""Build the LLMKT dataset: one prompt per labeled response.

Prompt layout (chat template, thinking disabled):
  system: task definition
  user:   transcript tail (token-truncated) + learning objective + question
The assistant's first token is the prediction; training reads the logits of
" True"/" False" (verified single-token at build time) at that position.

Transcript truncation is done here, once per session, by tokenizing the
transcript body alone and keeping the LAST MAX_TR tokens -- the end of the
session is closest to the quiz. The objective text goes AFTER the transcript
so it can never be truncated away.

Output: kt_data.jsonl with response_id, session_id, obj_fold, clus_fold, y,
prompt (full templated string ready for tokenization with add_special=False).
"""

import csv
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
from training_paths import DATA, TRANSCRIPTS
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-2B"
OUT = sys.argv[2] if len(sys.argv) > 2 else str(DATA / "kt_data.jsonl")
MAX_TR = 3400          # transcript tokens kept (tail)

SYSTEM = (
    "You are an expert mathematics tutor evaluator. You will read the end of "
    "a tutoring session transcript, then judge whether the student will "
    "answer a quiz question on the given learning objective correctly, "
    "immediately after the session. Respond with exactly one word: True if "
    "the student will answer correctly, False otherwise.")


def build_prompt(tok, transcript, objective):
    user = (f"Transcript (end of session):\n{transcript}\n\n"
            f"Learning objective of the quiz question:\n{objective.strip()}\n\n"
            "Will the student answer the quiz question correctly? "
            "Answer True or False.")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    if p.rstrip().endswith("<think>"):
        p = p.rstrip()[: -len("<think>")] + "<think>\n\n</think>\n\n"
    return p


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    for w in (" True", " False", "True", "False"):
        ids = tok.encode(w, add_special_tokens=False)
        print(f"token check {w!r}: {ids}")

    labels = {r["response_id"]: r for r in csv.DictReader(
        open(DATA / "folds.csv"))}
    feats = {r["response_id"]: r for r in csv.DictReader(
        open(DATA / "train_features_TMQTWsB.csv"))}

    # token-truncate each session transcript once
    sessions = sorted({r["session_id"] for r in labels.values()})
    print(f"{len(labels)} rows, {len(sessions)} sessions", flush=True)
    t0 = time.time()
    tail = {}
    for i, s in enumerate(sessions):
        with open(TRANSCRIPTS / f"{s}.csv", encoding="utf-8-sig",
                  newline="") as f:
            lines = [f"{(x.get('role') or '').strip()}: {x.get('content') or ''}"
                     for x in csv.DictReader(f)]
        body = "\n".join(lines)
        ids = tok.encode(body, add_special_tokens=False)
        if len(ids) > MAX_TR:
            body = tok.decode(ids[-MAX_TR:])
        tail[s] = body
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} sessions [{time.time() - t0:.0f}s]", flush=True)

    n_out = 0
    with open(BASE / OUT, "w") as out:
        for rid, r in labels.items():
            s = r["session_id"]
            if s not in tail:
                continue
            prompt = build_prompt(tok, tail[s],
                                  feats[rid]["learning_objective"])
            out.write(json.dumps({
                "response_id": rid, "session_id": s,
                "obj_fold": int(r["obj_fold"]),
                "clus_fold": int(r["clus_fold"]),
                "y": int(r["y"]), "prompt": prompt}) + "\n")
            n_out += 1
    print(f"wrote {n_out} examples  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
