"""LLMKT-style fine-tune of Qwen3.5-9B (Scarlatos et al., LAK 2025, sec 4.2).

The model reads a transcript tail + learning objective; we take the logits of
the tokens "True"/"False" at the first generated position, softmax over the
PAIR, and train with binary cross-entropy against the real quiz outcome.
This is exactly the competition loss, so the head is calibrated by
construction. No generation, no reports, no digit scores.

Train pool: obj_fold != 0 AND clus_fold != 0   (21.5k rows)
Eval each half epoch: obj_fold == 0 AND clus_fold == 0 (1,585 double-blocked
rows -- neither the objective nor the style cluster was seen in training).
Adapters saved per eval point; predictions dumped for the local harness.

Usage: python kt_train.py [--epochs 3] [--out runs/kt_v1]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
MICRO = 8              # overridden by --micro; ACCUM keeps eff. batch 64
ACCUM = 8
LR = 2e-4
MAX_LEN = 4096


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rows(data, split="folds"):
    rows = [json.loads(x) for x in open(Path(data))]
    if split == "all":
        # final retrain: everything except a 5% session holdout kept for
        # refitting temperature/blend on the shipped model
        import zlib
        hold = lambda r: zlib.crc32(r["session_id"].encode()) % 20 == 0
        return [r for r in rows if not hold(r)], [r for r in rows if hold(r)]
    train = [r for r in rows if r["obj_fold"] != 0 and r["clus_fold"] != 0]
    dblock = [r for r in rows if r["obj_fold"] == 0 and r["clus_fold"] == 0]
    return train, dblock


def batches(rows, tok, micro, shuffle, seed=0):
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    else:                      # sort by length for fast inference
        idx.sort(key=lambda i: len(rows[i]["prompt"]))
    for i in range(0, len(idx), micro):
        chunk = [rows[j] for j in idx[i:i + micro]]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAX_LEN,
                  add_special_tokens=False)
        y = torch.tensor([float(r["y"]) for r in chunk])
        yield chunk, enc, y


def pair_logits(model, enc, id_t, id_f):
    """LM head on the final position only (full-vocab logits for all 4096
    positions cost ~10GB per micro-batch). LoRA lives in the backbone, so
    gradients are unaffected; lm_head has no adapter."""
    base = model.get_base_model() if hasattr(model, "get_base_model") \
        else model
    h = base.model(input_ids=enc["input_ids"].cuda(),
                   attention_mask=enc["attention_mask"].cuda()
                   ).last_hidden_state
    last = (enc["attention_mask"].sum(1) - 1).cuda()
    hh = h[torch.arange(len(last), device=h.device), last]
    lg = base.get_output_embeddings()(hh)
    return lg[:, [id_t, id_f]].float()


def p_true(pair):
    return torch.softmax(pair, dim=-1)[:, 0]


@torch.no_grad()
def evaluate(model, tok, rows, id_t, id_f):
    model.eval()
    ps, ys, rids = [], [], []
    for chunk, enc, y in batches(rows, tok, MICRO * 2, shuffle=False):
        ps.append(p_true(pair_logits(model, enc, id_t, id_f)).cpu())
        ys.append(y)
        rids += [r["response_id"] for r in chunk]
    model.train()
    p = torch.cat(ps).numpy()
    y = torch.cat(ys).numpy()
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    llm = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    q = y.mean()
    llc = float(-(q * np.log(q) + (1 - q) * np.log(1 - q)))
    # AUC without sklearn: rank-sum
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    return llm, llc, float(auc), p, y, rids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="runs/kt_v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", default="kt_data.jsonl")
    ap.add_argument("--split", default="folds", choices=["folds", "all"])
    ap.add_argument("--micro", type=int, default=8)
    args = ap.parse_args()
    global MODEL, MICRO, ACCUM
    MODEL = args.model
    MICRO = args.micro
    ACCUM = 64 // MICRO
    out = BASE / args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    id_t = tok.encode("True", add_special_tokens=False)
    id_f = tok.encode("False", add_special_tokens=False)
    assert len(id_t) == 1 and len(id_f) == 1, (id_t, id_f)
    id_t, id_f = id_t[0], id_f[0]
    log(f"answer token ids: True={id_t} False={id_f}")

    train, dblock = load_rows(args.data, args.split)
    log(f"train {len(train)}  eval {len(dblock)}")
    if not train or not dblock:
        raise ValueError("KT training requires nonempty train and validation sets; supply more sessions or use --split folds")

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
        micro_iter = batches(train, tok, MICRO, shuffle=True, seed=ep)
        done = False
        while not done:
            opt.zero_grad(set_to_none=True)
            got = 0
            for _ in range(ACCUM):
                nxt = next(micro_iter, None)
                if nxt is None:
                    done = True
                    break
                _, enc, y = nxt
                pair = pair_logits(model, enc, id_t, id_f)
                lp = F.log_softmax(pair, dim=-1)
                loss = -(y.cuda() * lp[:, 0] +
                         (1 - y.cuda()) * lp[:, 1]).mean()
                (loss / ACCUM).backward()
                got += 1
                ema = loss.item() if ema is None else \
                    0.98 * ema + 0.02 * loss.item()
            if got == 0:
                break
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                log(f"ep{ep} step {step}/{total} loss_ema {ema:.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} "
                    f"[{(time.time() - t0) / 60:.1f}m]")
            if step % eval_every == 0 or step == total or done:
                llm, llc, auc, p, yv, rids = evaluate(
                    model, tok, dblock, id_t, id_f)
                log(f"EVAL step {step}: LL {llm:.5f} const {llc:.5f} "
                    f"dLL {llm - llc:+.5f} AUC {auc:.4f}")
                ck = out / f"ckpt-{step}"
                model.save_pretrained(ck)
                tok.save_pretrained(ck)
                with open(ck / "preds_dblock.csv", "w") as f:
                    f.write("response_id,p,y\n")
                    for r, pi, yi in zip(rids, p, yv):
                        f.write(f"{r},{pi:.6f},{int(yi)}\n")
    log("done")


if __name__ == "__main__":
    sys.exit(main())
