#!/usr/bin/env python
r"""Score (transcript, objective) rows with Transformers/PEFT -> P(correct); never imports vLLM.

The counterpart of score_vllm.py for readers vLLM 0.21 cannot serve as trained: Qwen3.5 trains
as the text-only Qwen3_5ForCausalLM, which vLLM 0.21 does not register. It shares score_vllm.py's
prompt building, training-recipe adoption, Yes/No token handling, row order and output
(response_id, p_llm). P(Yes) comes from a forward pass over the full vocabulary, so no row falls
back to 0.5 for want of a Yes/No token in a top-k list; rows are scored one at a time, so
logits_to_keep=1 is exact.

    python -m ace.llm.score_hf --model Qwen/Qwen3.5-9B \
        --adapter outputs/models/e23_qwen35_9b/fold0/adapter --data-dir data \
        --transcripts-dir data/train_transcripts --folds outputs/eda/cv_folds_skill.csv \
        --fold 0 --out outputs/models/e23_qwen35_9b/p_fold0.csv
"""
from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ace.llm.model_family import QWEN35, detect_family, load_causal_lm
from ace.llm.objective_context import build_prompt_messages, load_context_index
from ace.llm.serialize import answer_token_ids, apply_template, read_dialogue
# Reused, not reimplemented: a second copy of the recipe rules would drift, and the
# whole point of the guard is that training and scoring cannot silently disagree.
# score_vllm.py imports vllm lazily inside main(), so this import stays vLLM-free.
from ace.llm.score_vllm import RECIPE_DEFAULTS, find_training_config, resolve_prompt_recipe


def yes_probability(logits, yes_ids, no_ids) -> float:
    """P(Yes) from one position's vocabulary logits, matching score_vllm.py.

    Softmax over the full vocabulary, best Yes and best No variant, renormalised over the pair.
    score_vllm.py reads the same probabilities from a top-k logprob dict; here the whole
    distribution is available, so its py + pn <= 0 fallback to 0.5 cannot occur.
    """
    import torch
    probs = torch.softmax(logits.float(), dim=-1)
    py = float(max(probs[i].item() for i in yes_ids))
    pn = float(max(probs[i].item() for i in no_ids))
    if py + pn <= 0.0:  # unreachable with a real softmax; keeps parity with score_vllm
        return 0.5
    return py / (py + pn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="merged model dir or HF id, or (with --adapter) the base model")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir attached with PeftModel.from_pretrained; "
                         "tokenizer is read from the adapter dir when present")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--folds", default="outputs/eda/cv_folds_skill.csv",
                    help="CV folds csv (response_id, fold)")
    ap.add_argument("--fold", type=int, default=-1,
                    help="score only this fold's validation rows (-1 = all rows)")
    ap.add_argument("--out", required=True)
    # Recipe flags default to None = "adopt the trained value" (identical semantics to
    # score_vllm.py; RECIPE_DEFAULTS supplies the pre-knob fallback).
    ap.add_argument("--max-input-tokens", type=int, default=None, choices=[4096, 8192, 16384])
    ap.add_argument("--serialize", type=int, default=None, choices=[1, 2])
    ap.add_argument("--prompt-mode", default=None, choices=["standard", "objective_chunks"])
    ap.add_argument("--context-index", default=None)
    ap.add_argument("--require-context-index", action="store_true")
    ap.add_argument("--context-full-threshold", type=int, default=None)
    ap.add_argument("--context-chunks", type=int, default=None)
    ap.add_argument("--context-head-frac", type=float, default=None)
    ap.add_argument("--context-tail-frac", type=float, default=None)
    ap.add_argument("--allow-recipe-mismatch", action="store_true",
                    help="score with prompt settings that contradict the model's "
                         "training config (ablations only; normally an error)")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="deterministic row cap for compatibility smoke tests; 0 = all")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    config_path, cfg = find_training_config(args.model, args.adapter)
    notes = resolve_prompt_recipe(args, cfg, config_path)
    if config_path is not None:
        print(f"[score-hf] prompt recipe from {config_path}")
        for note in notes:
            print(f"[score-hf]   {note}")
    else:
        print("[score-hf] no training config found next to the model — using CLI/default "
              "prompt recipe (expected for untrained base-model runs)")
    print("[score-hf] recipe: " + " ".join(
        f"{k}={getattr(args, k)!r}" for k in RECIPE_DEFAULTS))

    data = Path(args.data_dir).expanduser()
    tdir = Path(args.transcripts_dir).expanduser()
    df = pd.read_csv(data / "train_features.csv")
    df["learning_objective"] = df["learning_objective"].str.strip()
    if args.fold >= 0:
        folds = pd.read_csv(args.folds)
        df = df.merge(folds, on="response_id", validate="1:1")
        df = df[df["fold"] == args.fold].reset_index(drop=True)
    # Same deterministic order as score_vllm.py. The HF backend has no prefix cache to
    # benefit from it, but keeping the order identical means a row-by-row diff against
    # a vLLM-scored CSV lines up, and --max-rows selects the same slice in both.
    df = df.sort_values(["session_id", "response_id"], kind="stable").reset_index(drop=True)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    print(f"[score-hf] {len(df):,} rows, model={args.model}"
          + (f", adapter={args.adapter}" if args.adapter else ""))

    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    yes_ids, no_ids = answer_token_ids(tok)
    print(f"[score-hf] yes ids={yes_ids} no ids={no_ids}")

    print("[score-hf] reading transcripts ...")
    sids = sorted(df["session_id"].unique())
    with Pool(args.workers) as pool:
        dialogues = dict(zip(sids, pool.map(partial(read_dialogue, version=args.serialize),
                                            [str(tdir / f"{s}.csv") for s in sids],
                                            chunksize=64)))

    context_index = None
    if args.context_index:
        context_index = load_context_index(args.context_index)
        bad = next((p for p in context_index.values() if p.serialize != args.serialize), None)
        if bad is not None:
            raise ValueError(
                f"context index serialize={bad.serialize} but --serialize={args.serialize}")
        print(f"[score-hf] loaded objective context index: {len(context_index):,} rows")
    if args.prompt_mode == "objective_chunks" and args.require_context_index:
        if not context_index:
            raise ValueError("objective_chunks with --require-context-index needs --context-index")
        missing = set(df["response_id"].astype(str)) - set(context_index)
        if missing:
            sample = sorted(missing)[:3]
            raise ValueError(f"required context index is missing {len(missing):,} "
                             f"scored rows; sample={sample}")

    print("[score-hf] building prompts ...")
    prompts = [apply_template(tok, build_prompt_messages(
                   tok, dialogues[s], o, max_input_tokens=args.max_input_tokens,
                   prompt_mode=args.prompt_mode, response_id=str(rid),
                   session_id=str(s), context_index=context_index,
                   require_context_index=args.require_context_index,
                   context_full_threshold=args.context_full_threshold,
                   context_chunks=args.context_chunks,
                   context_head_frac=args.context_head_frac,
                   context_tail_frac=args.context_tail_frac))
               for rid, s, o in zip(df["response_id"], df["session_id"],
                                    df["learning_objective"])]

    family, _ = detect_family(args.model)
    model, provenance = load_causal_lm(args.model, family)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(args.adapter).expanduser()))
        print(f"[score-hf] attached adapter {args.adapter}")
    model.config.use_cache = True  # scoring only: no gradient checkpointing in play
    model.eval()
    print(f"[score-hf] {provenance['model_class']} "
          f"(family={provenance['model_family']}, attn={provenance['attn_implementation']})")

    device = next(model.parameters()).device
    # batch=1: the prompt is the whole tensor, so the answer position is the last one
    # and logits_to_keep=1 gives exactly the scored logit.
    last_token_ok = family == QWEN35
    p_yes = np.full(len(df), np.nan)
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            ids = torch.tensor([list(prompt)], dtype=torch.long, device=device)
            att = torch.ones_like(ids)
            if last_token_ok:
                out = model(input_ids=ids, attention_mask=att, logits_to_keep=1)
            else:
                out = model(input_ids=ids, attention_mask=att)
            p_yes[i] = yes_probability(out.logits[0, -1, :], yes_ids, no_ids)
            if (i + 1) % 500 == 0:
                print(f"[score-hf]   {i + 1:,}/{len(df):,} rows")

    if not np.isfinite(p_yes).all():
        raise RuntimeError("non-finite probabilities produced — refusing to write")

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"response_id": df["response_id"], "p_llm": p_yes}).to_csv(out_path, index=False)
    print(f"[score-hf] wrote {out_path}")

    labels = data / "train_labels.csv"
    if labels.exists():
        from sklearn.metrics import roc_auc_score
        from ace.utils import logloss
        m = pd.read_csv(labels).merge(
            pd.DataFrame({"response_id": df["response_id"], "p": p_yes}),
            on="response_id")
        if m["is_correct"].nunique() > 1:
            print(f"[score-hf] raw ll={logloss(m['is_correct'], m['p']):.4f} "
                  f"auc={roc_auc_score(m['is_correct'], m['p']):.4f} "
                  f"(raw = uncalibrated; Platt/blend under protocol v3 comes next)")


if __name__ == "__main__":
    main()