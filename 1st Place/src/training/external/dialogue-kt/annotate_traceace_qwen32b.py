"""Self-annotate Trace the Ace transcripts with per-turn correctness pseudo-labels (HF generate).

Uses Qwen3-32B with the dialogue-kt correctness prompt.

Competition sessions are long (median ~102 LLMKT turns vs MathDial's ~5), so each session is
split into contiguous chunks of at most CHUNK turns, annotated independently (turns numbered
1..K within a chunk, mapped back to global indices). The per-turn summary chain-of-thought of
the dialogue-kt prompt is kept.

Output (append mode, resumable): results/traceace_turn_labels.csv with columns
session_id, turn (global 1-based LLMKT turn index), label in {true, false, na}.

Environment: ANNO_N_SESSIONS=1000, ANNO_CHUNK_TURNS=30, ANNO_BATCH=8, ANNO_OUT.

The pipeline does not run this script: step 02a uses annotate_traceace_vllm.py, the vLLM
version, which imports `chunk_prompts` from here (same chunking and prompts). The
annotator weights never ship. To run it, use the .env environment (transformers 4.51 supports
Qwen3-32B) on an 80 GB card.
"""
import os, sys, random
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.annotate import extract_result
from dialogue_kt.prompting import ANNO_CORRECTNESS_SYSTEM_PROMPT, TRACEACE_DIALOGUE_DESC
from build_traceace_kt import build_session_dialogue, find_one, MAGNIFICAT

MODEL = "Qwen/Qwen3-32B"
N_SESSIONS = int(os.environ.get("ANNO_N_SESSIONS", "1000"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "30"))
BATCH = int(os.environ.get("ANNO_BATCH", "8"))
OUT = os.environ.get("ANNO_OUT", "results/traceace_turn_labels.csv")
SEED = 221
DATA = f"{MAGNIFICAT}/data"

from transformers import AutoTokenizer, AutoModelForCausalLM


def chunk_prompts(sid: str, turns: list):
    """Split a session into contiguous user prompts of at most CHUNK turns.

    Args:
        sid: session id.
        turns: the session's turn list from ``build_session_dialogue``.

    Yields:
        ``((sid, start, k), prompt)`` with ``start`` the 0-based offset of the chunk and ``k``
        its number of turns (numbered 1..k in the prompt).
    """
    for start in range(0, len(turns), CHUNK):
        chunk = turns[start:start + CHUNK]
        lines = []
        for i, t in enumerate(chunk, start=1):
            if t["teacher"]:
                lines.append(f"Teacher Turn {i}: {t['teacher']}")
            if t["student"]:
                lines.append(f"Student Turn {i}: {t['student']}")
        prompt = "[BEGIN DIALOGUE]\n" + "\n".join(lines) + "\n[END DIALOGUE]"
        prompt += f"\n\nYour final response should have an entry for exactly {len(chunk)} turns (1-{len(chunk)})."
        yield (sid, start, len(chunk)), prompt


def main():
    """Annotate an objective-stratified sample of sessions and append the labels to OUT.

    Returns:
        None. Appends ``session_id, turn, label`` rows to OUT.
    """
    feats = pd.read_csv(find_one(f"{DATA}/train_features*.csv"))
    # LO-stratified session sample: 1 session per LO first, then fill randomly
    rng = random.Random(SEED)
    first = feats.groupby("learning_objective_id", group_keys=False).head(1)
    sids = list(dict.fromkeys(first["session_id"]))
    rest = [s for s in feats["session_id"].unique() if s not in set(sids)]
    rng.shuffle(rest)
    sids = (sids + rest)[:N_SESSIONS]

    done = set()
    if os.path.exists(OUT):  # resume: skip fully-annotated sessions
        done = set(pd.read_csv(OUT)["session_id"].unique())
        sids = [s for s in sids if s not in done]
        print(f"resume: {len(done)} sessions already annotated, {len(sids)} to go", flush=True)

    system = ANNO_CORRECTNESS_SYSTEM_PROMPT.format(desc=TRACEACE_DIALOGUE_DESC)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    header_needed = not os.path.exists(OUT)
    n_parse_fail = 0
    buf_meta, buf_prompt = [], []

    def flush():
        nonlocal n_parse_fail, header_needed
        if not buf_prompt:
            return
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in buf_prompt]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=2048, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        rows = []
        for j, (sid, start, k) in enumerate(buf_meta):
            raw = tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_result(raw)
            if not isinstance(pred, dict):
                n_parse_fail += 1
                continue
            for key, val in pred.items():
                try:
                    local = int(str(key).lower().replace("turn", "").strip())
                except ValueError:
                    continue
                if 1 <= local <= k and str(val).lower().strip() in ("true", "false", "na"):
                    rows.append({"session_id": sid, "turn": start + local,
                                 "label": str(val).lower().strip()})
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=header_needed, index=False)
            header_needed = False
        buf_meta.clear(); buf_prompt.clear()

    n_chunks = 0
    for si, sid in enumerate(sids):
        path = f"{DATA}/train_transcripts/{sid}.csv"
        if not os.path.exists(path):
            continue
        turns = build_session_dialogue(pd.read_csv(path))
        if not turns:
            continue
        for meta, prompt in chunk_prompts(sid, turns):
            buf_meta.append(meta); buf_prompt.append(prompt)
            n_chunks += 1
            if len(buf_prompt) >= BATCH:
                flush()
        if (si + 1) % 20 == 0:
            print(f"  sessions {si + 1}/{len(sids)}  chunks {n_chunks}  parse_fail {n_parse_fail}", flush=True)
    flush()
    print(f"DONE sessions={len(sids)} chunks={n_chunks} parse_fail={n_parse_fail} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
