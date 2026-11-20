"""Train Qwen3.5-2B on human per-exchange correctness annotations.

Three-way cross-entropy over True, False, and NA. Human verdicts come from
DATA/turn_labels.csv; quiz outcomes are never used as exchange verdicts.
Commands: build (prompts), train (LoRA), label (probabilities for all pairs).
"""

import argparse
import csv
import json
import math
import random
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
from training_paths import DATA, RUNS
MODEL = str(BASE / "models" / "judge2b_base")
CTX = 3                 # preceding pairs shown as context
T_CUT, S_CUT = 600, 400
MAX_LEN = 1024
VERDICTS = ["true", "false", "na"]

SYSTEM = (
    "You are an experienced math tutor reviewing a transcribed SPOKEN "
    "dialogue between a tutor and a student in an online math tutoring "
    "session. Transcripts contain filler words and [unclear] where audio was "
    "lost, and speaker attribution is occasionally imperfect. Judge whether "
    "the student correctly answered the tutor's question in the FINAL "
    "exchange. Respond with exactly one word: True if the student's answer "
    "to an assessing question is correct; False if the tutor asked an "
    "assessing question with a correct answer and the student answered "
    "incorrectly or not at all; NA if the tutor's turn does not assess "
    "mathematical knowledge, has no right or wrong answer, or the student "
    "turn is too garbled to judge.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def answer_ids(tok):
    ids = []
    for w in ("True", "False", "NA"):
        e = tok.encode(w, add_special_tokens=False)
        assert len(e) == 1, (w, e)
        ids.append(e[0])
    return ids


def build_prompt(tok, ctx_pairs, pair):
    lines = []
    for p in ctx_pairs:
        if p["tutor"]:
            lines.append(f"Tutor: {p['tutor'][:T_CUT]}")
        lines.append(f"Student: {p['student'][:S_CUT]}")
    lines.append("--- FINAL EXCHANGE ---")
    if pair["tutor"]:
        lines.append(f"Tutor: {pair['tutor'][:T_CUT]}")
    lines.append(f"Student: {pair['student'][:S_CUT]}")
    user = ("\n".join(lines) +
            "\n\nDid the student answer the tutor's question in the final "
            "exchange correctly? Answer True, False, or NA.")
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


def load_pairs(sessions):
    by = defaultdict(dict)
    for line in open(DATA / "pairs.jsonl"):
        p = json.loads(line)
        if p["session_id"] in sessions:
            by[p["session_id"]][p["pair_idx"]] = p
    return by


def cmd_build(_):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    log(f"answer token ids: {answer_ids(tok)}")

    from prepare_training import validated_verdicts
    verd = defaultdict(dict)
    for (session, index), verdict in validated_verdicts().items():
        verd[session][index] = verdict
    sessions = sorted(verd)
    # Group by session, including a deterministic fallback for small annotation sets.
    val_sessions = {s for s in sessions if zlib.crc32(s.encode()) % 10 == 0}
    if not val_sessions or len(val_sessions) == len(sessions):
        val_sessions = {sessions[0]}
    pairs = load_pairs(set(verd))
    log(f"{len(verd)} sessions, "
        f"{sum(len(v) for v in verd.values())} labeled pairs")

    t0 = time.time()
    n = {"train": 0, "val": 0}
    with open(DATA / "judge_data.jsonl", "w") as out:
        for s in sorted(verd):
            split = "val" if s in val_sessions else "train"
            pd = pairs[s]
            for i, v in sorted(verd[s].items()):
                if i not in pd:
                    continue
                ctx = [pd[j] for j in range(max(i - CTX, 0), i) if j in pd]
                prompt = build_prompt(tok, ctx, pd[i])
                out.write(json.dumps({
                    "session_id": s, "pair_idx": i, "split": split,
                    "v": VERDICTS.index(v), "prompt": prompt}) + "\n")
                n[split] += 1
    log(f"wrote {n['train']} train + {n['val']} val  "
        f"[{time.time() - t0:.0f}s]")


def batches(rows, tok, micro, shuffle, seed=0):
    import torch
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    else:
        idx.sort(key=lambda i: len(rows[i]["prompt"]))
    for i in range(0, len(idx), micro):
        chunk = [rows[j] for j in idx[i:i + micro]]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False)
        y = torch.tensor([r.get("v", 0) for r in chunk])
        yield chunk, enc, y


def tri_logits(model, enc, ids):
    """LM head applied to the final position only -- full-vocab logits for
    every position OOM at inference batch sizes."""
    import torch
    model = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = model.model(input_ids=enc["input_ids"].cuda(),
                    attention_mask=enc["attention_mask"].cuda()
                    ).last_hidden_state
    last = (enc["attention_mask"].sum(1) - 1).cuda()
    hh = h[torch.arange(len(last), device=h.device), last]
    return model.lm_head(hh)[:, ids].float()


def evaluate(model, tok, rows, ids, micro):
    import numpy as np
    import torch
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for _, enc, y in batches(rows, tok, micro, shuffle=False):
            P.append(torch.softmax(tri_logits(model, enc, ids), -1).cpu())
            Y.append(y)
    model.train()
    P = torch.cat(P).numpy()
    Y = torch.cat(Y).numpy()
    pred = P.argmax(1)
    acc = float((pred == Y).mean())
    ce = float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-7, 1)).mean())
    lines = [f"acc {acc:.4f}  ce {ce:.4f}"]
    for k, name in enumerate(VERDICTS):
        m = Y == k
        if m.any():
            rec = float((pred[m] == k).mean())
            prec = float((Y[pred == k] == k).mean()) if (pred == k).any() else 0
            lines.append(f"{name}: n={m.sum()} recall {rec:.3f} prec {prec:.3f}")
    return acc, ce, "  |  ".join(lines)


def cmd_train(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MICRO, ACCUM, LR = 16, 4, 2e-4
    out = RUNS / "judge"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = answer_ids(tok)
    rows = [json.loads(x) for x in open(DATA / "judge_data.jsonl")]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    log(f"train {len(train)}  val {len(val)}")
    if not train or not val:
        raise ValueError("Judge training requires nonempty train and validation sets")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lcfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lcfg).cuda()
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train) / (MICRO * ACCUM))
    total = steps_per_epoch * args.epochs
    warm = max(int(0.03 * total), 10)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm) * 0.5 *
        (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(total - warm, 1)))))
    eval_every = max(steps_per_epoch // 2, 1)
    log(f"{steps_per_epoch} steps/epoch, {total} total, eval every {eval_every}")

    step, ema, t0 = 0, None, time.time()
    model.train()
    for ep in range(args.epochs):
        it = batches(train, tok, MICRO, shuffle=True, seed=ep)
        done = False
        while not done:
            opt.zero_grad(set_to_none=True)
            got = 0
            for _ in range(ACCUM):
                nxt = next(it, None)
                if nxt is None:
                    done = True
                    break
                _, enc, y = nxt
                loss = F.cross_entropy(tri_logits(model, enc, ids), y.cuda())
                (loss / ACCUM).backward()
                got += 1
                ema = loss.item() if ema is None else \
                    0.98 * ema + 0.02 * loss.item()
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                log(f"ep{ep} step {step}/{total} loss_ema {ema:.4f} "
                    f"[{(time.time() - t0) / 60:.1f}m]")
            if step % eval_every == 0 or step == total or done:
                _, _, rep = evaluate(model, tok, val, ids, MICRO * 2)
                log(f"EVAL step {step}: {rep}")
                model.save_pretrained(str(out / f"ckpt-{step}"))
                tok.save_pretrained(str(out / f"ckpt-{step}"))
                (out / "latest_checkpoint.txt").write_text(str((out / f"ckpt-{step}").resolve()))
    log("done")


def cmd_label(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MICRO = 64
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = answer_ids(tok)

    todo = defaultdict(list)          # session -> [pair_idx] (assessable)
    for r in csv.DictReader(open(DATA / "assign.csv")):
        todo[r["session_id"]].append(int(r["pair_idx"]))
    log(f"{sum(len(v) for v in todo.values())} assessable pairs over {len(todo)} sessions")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(Path(args.ckpt).resolve()))
    model = model.merge_and_unload().cuda()
    model.eval()

    t0 = time.time()
    n_done = 0
    sessions = sorted(todo)
    fout = open(DATA / "pair_probs_qwen_all.csv", "w", newline="")
    w = csv.writer(fout)
    w.writerow(["session_id", "pair_idx", "p_true", "p_false", "p_na"])
    # stream sessions in blocks to bound memory
    BLOCK = 400
    for b0 in range(0, len(sessions), BLOCK):
        blk = set(sessions[b0:b0 + BLOCK])
        pairs = load_pairs(blk)
        rows = []
        for s in sorted(blk):
            pd = pairs[s]
            for i in sorted(todo[s]):
                if i not in pd:
                    continue
                ctx = [pd[j] for j in range(max(i - CTX, 0), i) if j in pd]
                rows.append({"session_id": s, "pair_idx": i,
                             "prompt": build_prompt(tok, ctx, pd[i])})
        with torch.no_grad():
            for chunk, enc, _ in batches(rows, tok, MICRO, shuffle=False):
                P = torch.softmax(
                    tri_logits(model, enc, ids), -1).cpu().numpy()
                for r, p in zip(chunk, P):
                    w.writerow([r["session_id"], r["pair_idx"],
                                f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
        n_done += len(rows)
        fout.flush()
        log(f"{n_done} pairs labeled  [{(time.time() - t0) / 60:.1f}m]")
    fout.close()
    log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "label"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--ckpt", required=False)
    args = ap.parse_args()
    if args.cmd == "label" and not args.ckpt:
        ap.error("label requires --ckpt")
    {"build": cmd_build, "train": cmd_train, "label": cmd_label}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
