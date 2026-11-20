#!/usr/bin/env python
r"""LoRA fine-tune of a causal LM as a Yes/No transcript reader (Qwen3-4B/8B, Qwen3.5-9B).

The target is the single answer token after the serialised (transcript, objective) prompt;
the loss is masked everywhere else, and at inference the probability is the softmax over the
Yes/No logits (score_vllm.py). Folds are the skill-cluster folds, and training rows whose
session also appears in the held-out fold are dropped. --fold -1 trains the full-fit model.
--synth-dir adds synthetic rows to the training side of every fold (never validation), on
the generator's soft label by default; --synth-purge-csv (scripts/gen_synth/make_purge.py)
drops rows whose objective is a near-twin of the held-out fold's objectives.

    python -m ace.llm.train_lora --model Qwen/Qwen3-4B --data-dir data \
        --transcripts-dir data/train_transcripts --folds outputs/eda/cv_folds_skill.csv \
        --fold 0 --out outputs/models/e13_qwen4/fold0
"""
from __future__ import annotations

import argparse
import json
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ace.llm.model_family import (QWEN35, can_use_last_token_logits, detect_family,
                                  first_supported_kwarg, load_causal_lm)
from ace.llm.objective_context import build_prompt_messages, load_context_index
from ace.llm.serialize import apply_template, read_dialogue


def build_dataset(df, dialogues, tok, max_input_tokens, *, prompt_mode="standard",
                  context_index=None, require_context_index=False,
                  context_full_threshold=0, context_chunks=2,
                  context_head_frac=0.16, context_tail_frac=0.34):
    """Rows carry a target p = P(Yes) (real rows 0/1, synthetic rows p_soft) and a loss weight w.

    The input is the prompt only; the answer token is scored from the logits at the last prompt
    position (same gradients as appending it, and the same input score_vllm.py feeds).
    """
    import torch
    from torch.utils.data import Dataset

    class Rows(Dataset):
        def __len__(self):
            return len(df)

        def __getitem__(self, i):
            r = df.iloc[i]
            messages = build_prompt_messages(
                tok, dialogues[r["session_id"]], r["learning_objective"],
                max_input_tokens=max_input_tokens, prompt_mode=prompt_mode,
                response_id=str(r.get("response_id", "")),
                session_id=str(r["session_id"]), context_index=context_index,
                require_context_index=require_context_index,
                context_full_threshold=context_full_threshold,
                context_chunks=context_chunks, context_head_frac=context_head_frac,
                context_tail_frac=context_tail_frac)
            prompt = apply_template(tok, messages)
            return {"input_ids": torch.tensor(list(prompt)),
                    "p": float(r["p"]), "w": float(r["w"]),
                    "length": len(prompt)}
    return Rows()


def collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = b["input_ids"]
        att[i, :k] = 1
    return {"input_ids": ids, "attention_mask": att,
            "ans_pos": torch.tensor([len(b["input_ids"]) - 1 for b in batch]),
            "p": torch.tensor([b["p"] for b in batch], dtype=torch.float),
            "w": torch.tensor([b["w"] for b in batch], dtype=torch.float)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--folds", default="outputs/eda/cv_folds_skill.csv")
    ap.add_argument("--fold", type=int, required=True,
                    help="held-out fold (not trained on); -1 = full fit on every row "
                         "(no validation, the synth purge does nothing): the deployment "
                         "model, which has no honest OOF, so the stack is fitted on the "
                         "fold models' predictions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-input-tokens", type=int, default=8192, choices=[4096, 8192, 16384])
    ap.add_argument("--serialize", type=int, default=1, choices=[1, 2],
                    help="dialogue serialization version (2 = diarization repair: "
                         "split on [Speaker:...] tags, keep substantial background; "
                         "score and deploy with the same version)")
    ap.add_argument("--prompt-mode", default="standard",
                    choices=["standard", "objective_chunks"],
                    help="standard prompt, or objective-aware head/chunks/tail prompt")
    ap.add_argument("--context-index", default=None,
                    help="JSONL semantic context index from prepare_objective_contexts")
    ap.add_argument("--require-context-index", action="store_true",
                    help="fail rather than use lexical fallback when a response is absent")
    ap.add_argument("--context-full-threshold", type=int, default=0,
                    help="keep sessions whole up to this many evidence tokens; 0 = use the full available evidence budget")
    ap.add_argument("--context-chunks", type=int, default=2)
    ap.add_argument("--context-head-frac", type=float, default=0.16)
    ap.add_argument("--context-tail-frac", type=float, default=0.34)
    ap.add_argument("--synth-dir", default=None,
                    help="synthetic-data dir (features.csv, labels.csv, transcripts/); "
                         "rows join the training side of every fold, never validation")
    ap.add_argument("--synth-soft", type=int, default=1, choices=[0, 1],
                    help="1: train synth rows on the generator's latent p_soft "
                         "(soft CE over the Yes/No token); 0: hard is_correct")
    ap.add_argument("--synth-weight", type=float, default=1.0,
                    help="loss weight for synth rows (real rows are 1.0)")
    ap.add_argument("--synth-purge-csv", default=None,
                    help="near-twin purge list from scripts/gen_synth/make_purge.py "
                         "(response_id,fold): synth rows dropped when their objective "
                         "is a twin of the held-out fold's validation objectives")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch size")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--merge", action="store_true",
                    help="also export the merged model to <out>/merged (ship this)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dataloader-workers", type=int, default=4,
                    help="set 0 if training hangs at step 0 (old-kernel warning: "
                         "kernels <5.5 can rarely hang multi-worker DataLoaders)")
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="deterministic training-row cap for compatibility smoke tests; 0 = all")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Trainer max_steps override for compatibility smoke tests; -1 = epochs")
    ap.add_argument("--seed", type=int, default=42,
                    help="LoRA init + data order seed (42 for every shipped model); change "
                         "it only for a deliberate seed ensemble, and record which seed "
                         "produced which adapter")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir).expanduser()
    tdir = Path(args.transcripts_dir).expanduser()

    df = pd.read_csv(data / "train_features.csv")
    df["learning_objective"] = df["learning_objective"].str.strip()
    df = df.merge(pd.read_csv(data / "train_labels.csv"), on="response_id", validate="1:1")
    df = df.merge(pd.read_csv(args.folds), on="response_id", validate="1:1")

    va = df["fold"] == args.fold
    va_sessions = set(df.loc[va, "session_id"])
    tr_df = df[(df["fold"] != args.fold)
               & ~df["session_id"].isin(va_sessions)].reset_index(drop=True)
    print(f"[train] fold {args.fold}: {len(tr_df):,} train rows "
          f"({len(df):,} total, {int(va.sum()):,} val, purge included)")

    tr_df["p"] = tr_df["is_correct"].astype(float)
    tr_df["w"] = 1.0
    if args.max_train_rows > 0 and len(tr_df) > args.max_train_rows:
        tr_df = (tr_df.sample(n=args.max_train_rows, random_state=42)
                       .sort_values("response_id", kind="stable")
                       .reset_index(drop=True))
        print(f"[train] compatibility cap: {len(tr_df):,} deterministic rows")
    n_synth = 0
    if args.synth_dir:
        sdir = Path(args.synth_dir).expanduser()
        syn = pd.read_csv(sdir / "features.csv")
        syn["learning_objective"] = syn["learning_objective"].str.strip()
        syn = syn.merge(pd.read_csv(sdir / "labels.csv"), on="response_id", validate="1:1")
        n_raw = len(syn)
        if args.synth_purge_csv:
            purge = pd.read_csv(args.synth_purge_csv)
            drop = set(purge.loc[purge["fold"] == args.fold, "response_id"])
            syn = syn[~syn["response_id"].isin(drop)].reset_index(drop=True)
        assert not (set(syn["session_id"]) & set(df["session_id"])), \
            "synth session_ids collide with real ones"
        syn["p"] = (syn["p_soft"] if args.synth_soft else syn["is_correct"]).astype(float)
        syn["w"] = args.synth_weight
        n_synth = len(syn)
        print(f"[train] synth: +{n_synth:,} rows from {sdir} "
              f"({n_raw - n_synth:,} purged for fold {args.fold}, "
              f"{'p_soft' if args.synth_soft else 'hard'} targets, w={args.synth_weight})")
        cols = ["response_id", "session_id", "learning_objective", "p", "w"]
        tr_df = pd.concat([tr_df[cols], syn[cols]], ignore_index=True)

    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    yes = tok.encode("Yes", add_special_tokens=False)
    no = tok.encode("No", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1, \
        "Yes/No must be single tokens for this tokenizer (see serialize.answer_token_ids)"

    print("[train] reading transcripts ...")
    src = {s: tdir for s in tr_df["session_id"].unique()}
    if args.synth_dir:
        stdir = Path(args.synth_dir).expanduser() / "transcripts"
        src.update({s: stdir for s in syn["session_id"].unique()})
    sids = sorted(src)
    with Pool(args.workers) as pool:
        dialogues = dict(zip(sids, pool.map(partial(read_dialogue, version=args.serialize),
                                            [str(src[s] / f"{s}.csv") for s in sids],
                                            chunksize=64)))
    context_index = None
    if args.context_index:
        context_index = load_context_index(args.context_index)
        bad = next((p for p in context_index.values() if p.serialize != args.serialize), None)
        if bad is not None:
            raise ValueError(
                f"context index serialize={bad.serialize} but --serialize={args.serialize}")
        print(f"[train] loaded objective context index: {len(context_index):,} rows")
    if args.prompt_mode == "objective_chunks" and args.require_context_index and not context_index:
        raise ValueError("objective_chunks with --require-context-index needs --context-index")
    if args.prompt_mode == "objective_chunks" and args.require_context_index:
        missing = set(tr_df["response_id"].astype(str)) - set(context_index or {})
        if missing:
            sample = sorted(missing)[:3]
            raise ValueError(f"required context index is missing {len(missing):,} training rows; sample={sample}")
    ds = build_dataset(
        tr_df, dialogues, tok, args.max_input_tokens, prompt_mode=args.prompt_mode,
        context_index=context_index, require_context_index=args.require_context_index,
        context_full_threshold=args.context_full_threshold,
        context_chunks=args.context_chunks, context_head_frac=args.context_head_frac,
        context_tail_frac=args.context_tail_frac)

    import torch
    # Family detection drives the load path: Qwen3.5 uses its text-only class with SDPA,
    # every other model flash-attn-2 when available, else SDPA.
    family, _model_config = detect_family(args.model)
    model, provenance = load_causal_lm(args.model, family)
    print(f"[train] {provenance['model_class']} "
          f"(family={provenance['model_family']}, attn={provenance['attn_implementation']})")
    model.config.use_cache = False
    # required with gradient checkpointing + frozen base: keeps grad flowing from the
    # embeddings into the checkpointed blocks (else: "element 0 does not require grad")
    model.enable_input_require_grads()
    # Seed before get_peft_model: PEFT draws the LoRA A matrices from the global torch RNG
    # here, and Trainer.__init__ only calls set_seed later, so passing the seed through
    # TrainingArguments alone would reshuffle the data order while leaving every adapter
    # identically initialised. A seed ensemble needs both.
    from transformers import set_seed
    set_seed(args.seed)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Two TrainingArguments knobs were renamed in Transformers 5 without changing meaning.
    # The same recipe must run on Transformers 4.57 (Qwen3) and 5 (Qwen3.5), so each is
    # resolved by capability, oldest spelling first: a build exposing both keeps the
    # spelling the Qwen3 runs used, so they stay reproducible.
    compat_targs = {}

    # 3% warmup. warmup_steps takes a float in [0, 1) as a ratio in Transformers 5,
    # so 0.03 means the same thing under either name. Never silently dropped: the
    # recipe depends on it, and a first optimizer step at ~0 LR is a real difference.
    warmup = first_supported_kwarg(
        TrainingArguments, [("warmup_ratio", 0.03), ("warmup_steps", 0.03)])
    if warmup is None:
        raise RuntimeError(
            "TrainingArguments accepts neither 'warmup_ratio' nor 'warmup_steps' in this "
            "Transformers build — ACE's recipe requires 3% warmup and will not train "
            "without it. Check the installed transformers version.")
    compat_targs[warmup[0]] = warmup[1]

    # Length grouping is not a no-op at batch=1: it also sets the order in which examples
    # are seen, which matters under gradient accumulation 16.
    sampling = first_supported_kwarg(
        TrainingArguments,
        [("group_by_length", True), ("train_sampling_strategy", "group_by_length")])
    if sampling is None:
        print("[train] WARNING: TrainingArguments accepts neither 'group_by_length' nor "
              "'train_sampling_strategy' — falling back to the default sampler; example "
              "ORDER will differ from the existing Qwen3 runs")
    else:
        compat_targs[sampling[0]] = sampling[1]
    print("[train] TrainingArguments compat: "
          + ", ".join(f"{k}={v!r}" for k, v in compat_targs.items()))

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr, lr_scheduler_type="cosine",
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **compat_targs,
        logging_steps=25, save_strategy="no", report_to=[], seed=args.seed,
        dataloader_num_workers=args.dataloader_workers, remove_unused_columns=False,
    )
    # Only Qwen3.5 opts into the single-position forward; the Qwen3 readers keep the
    # full-logits gather so their results stay bit-reproducible.
    use_last_token = family == QWEN35
    class SoftTrainer(Trainer):
        """Cross-entropy against the Yes/No token logits with target p = P(Yes) and row weight w.

        For real rows (p in {0, 1}, w = 1) this is exactly the masked-LM loss on the single answer
        token; synthetic rows may carry the generator's soft label.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            p, w, pos = inputs.pop("p"), inputs.pop("w"), inputs.pop("ans_pos")
            ids, att = inputs["input_ids"], inputs["attention_mask"]
            if use_last_token and can_use_last_token_logits(pos, ids.size(1)):
                # Every answer position is the last tensor position, so materialising
                # [B, L, V] to keep one row is pure waste (a 9B vocab x 8K context is
                # gigabytes); logits_to_keep=1 gives the same numbers for that position.
                out = model(input_ids=ids, attention_mask=att, logits_to_keep=1)
                ans_logits = out.logits[:, -1, :].float()
            else:
                out = model(input_ids=ids, attention_mask=att)
                idx = pos.view(-1, 1, 1).expand(-1, 1, out.logits.size(-1))
                ans_logits = out.logits.gather(1, idx).squeeze(1).float()
            logp = torch.log_softmax(ans_logits, dim=-1)
            loss = -(p * logp[:, yes[0]] + (1 - p) * logp[:, no[0]])
            loss = (w * loss).sum() / w.sum()
            return (loss, out) if return_outputs else loss

    trainer = SoftTrainer(model=model, args=targs, train_dataset=ds,
                          data_collator=lambda b: collate(b, pad_id))
    trainer.train()

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    if args.merge:
        print("[train] merging adapter for deployment ...")
        merged = model.merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")

    (out / "config.json").write_text(json.dumps(
        {"model": args.model, "fold": args.fold, "folds_csv": args.folds,
         "serialize": args.serialize, "prompt_mode": args.prompt_mode,
         "context_index": args.context_index,
         "require_context_index": args.require_context_index,
         "context_full_threshold": args.context_full_threshold,
         "context_chunks": args.context_chunks,
         "context_head_frac": args.context_head_frac,
         "context_tail_frac": args.context_tail_frac,
         "max_input_tokens": args.max_input_tokens, "epochs": args.epochs,
         "lr": args.lr, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
         "batch": args.batch, "accum": args.accum,
         "max_train_rows": args.max_train_rows, "max_steps": args.max_steps,
         "seed": args.seed,
         # Provenance: which class/kernel actually produced this adapter, and whether
         # the single-position forward was in play. An OOF file is only interpretable
         # alongside these.
         **provenance,
         "last_token_logits": bool(use_last_token),
         "synth_dir": args.synth_dir, "synth_rows": n_synth,
         "synth_soft": args.synth_soft, "synth_weight": args.synth_weight,
         "synth_purge_csv": args.synth_purge_csv,
         "train_rows": len(tr_df), "runtime_s": round(time.time() - t0)}, indent=2))
    print(f"[train] done in {(time.time()-t0)/3600:.1f} h -> {out}")


if __name__ == "__main__":
    main()