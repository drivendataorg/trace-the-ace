#!/usr/bin/env python
r"""Score (transcript, objective) rows with a causal LLM through vLLM -> P(correct).

Scores a merged model, or a base model plus a LoRA adapter (--adapter, served with vLLM
enable_lora as at inference). P(Yes) is read from the first generated token's logprobs over the
Yes/No variants, never from generated text; deterministic (temperature 0, one token).
Every setting that changes the prompt text (serialize, max_input_tokens, prompt_mode, context_*)
is read from the model's training config.json when one sits next to the adapter or merged dir.
Omitted flags adopt the trained values and contradicting flags are an error
(--allow-recipe-mismatch overrides): a different prompt gives silently wrong predictions.

    python -m ace.llm.score_vllm --model Qwen/Qwen3-4B --adapter outputs/models/e13_qwen4/fold0/adapter \
        --data-dir data --transcripts-dir data/train_transcripts --folds outputs/eda/cv_folds_skill.csv \
        --fold 0 --serialize 1 --out outputs/models/e13_qwen4/p_fold0.csv
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ace.llm.objective_context import build_prompt_messages, load_context_index
from ace.llm.serialize import answer_token_ids, apply_template, read_dialogue

# Prompt-affecting knobs, with the values assumed for a model whose training config
# predates that knob (the Qwen3-4B/8B fold runs).
RECIPE_DEFAULTS = {
    "serialize": 1,
    "max_input_tokens": 8192,
    "prompt_mode": "standard",
    "context_full_threshold": 0,
    "context_chunks": 2,
    "context_head_frac": 0.16,
    "context_tail_frac": 0.34,
}


def find_training_config(model: str, adapter: str | None) -> tuple[Path | None, dict]:
    """Locate the train_lora config.json for the model being scored.

    train_lora writes it to <out>/config.json, next to <out>/adapter and <out>/merged. A base
    model's own config.json must never be mistaken for it, so a candidate only counts when it
    carries training-recipe keys.
    """
    candidates = []
    if adapter:
        candidates.append(Path(adapter).expanduser().parent / "config.json")
    m = Path(model).expanduser()
    if m.is_dir():
        candidates.append(m.parent / "config.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            cfg = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(cfg, dict) and ("prompt_mode" in cfg or "lora_r" in cfg):
            return path, cfg
    return None, {}


def resolve_prompt_recipe(args, cfg: dict, config_path: Path | None) -> list[str]:
    """Fill unset recipe flags from the training config; reject contradictions."""
    notes, conflicts = [], []
    for field, fallback in RECIPE_DEFAULTS.items():
        trained = cfg.get(field, fallback) if config_path is not None else None
        given = getattr(args, field)
        if given is None:
            resolved = trained if config_path is not None else fallback
            setattr(args, field, resolved)
            if config_path is not None:
                notes.append(f"{field}={resolved!r} (from training config)")
        elif config_path is not None and given != trained:
            conflicts.append(f"  --{field.replace('_', '-')}: passed {given!r}, trained {trained!r}")

    # The index path is machine-specific, but training with semantic plans and
    # scoring on the lexical fallback (or vice versa) changes every prompt.
    if config_path is not None and args.prompt_mode == "objective_chunks":
        trained_idx = bool(cfg.get("context_index"))
        given_idx = bool(args.context_index)
        if trained_idx != given_idx:
            conflicts.append(
                f"  --context-index: {'supplied' if given_idx else 'absent'} now, "
                f"{'used' if trained_idx else 'not used'} in training")

    if conflicts and not args.allow_recipe_mismatch:
        raise SystemExit(
            f"[score] prompt recipe disagrees with {config_path}:\n"
            + "\n".join(conflicts)
            + "\nScoring with a different prompt than the model was trained on gives"
              " wrong predictions.\nOmit the flags to adopt the trained values, or pass"
              " --allow-recipe-mismatch if the difference is deliberate.")
    if conflicts:
        notes.append("MISMATCH ACCEPTED: " + "; ".join(x.strip() for x in conflicts))
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="merged model dir, HF id, or (with --adapter) the base model")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter dir, served with vLLM enable_lora as at inference "
                         "(no merged/ needed); the tokenizer is read from the adapter dir")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--folds", default="outputs/eda/cv_folds_skill.csv",
                    help="CV folds csv (response_id, fold)")
    ap.add_argument("--fold", type=int, default=-1,
                    help="score only this fold's validation rows (-1 = all rows)")
    ap.add_argument("--out", required=True)
    # Recipe flags default to None = "adopt the trained value" (see RECIPE_DEFAULTS
    # for the fallback used when no training config is found).
    ap.add_argument("--max-input-tokens", type=int, default=None, choices=[4096, 8192, 16384])
    ap.add_argument("--serialize", type=int, default=None, choices=[1, 2],
                    help="dialogue serialization version; must match the scored model's "
                         "training and is read from its config.json when omitted")
    ap.add_argument("--prompt-mode", default=None,
                    choices=["standard", "objective_chunks"])
    ap.add_argument("--context-index", default=None,
                    help="JSONL semantic context index from prepare_objective_contexts")
    ap.add_argument("--require-context-index", action="store_true")
    ap.add_argument("--context-full-threshold", type=int, default=None)
    ap.add_argument("--context-chunks", type=int, default=None)
    ap.add_argument("--context-head-frac", type=float, default=None)
    ap.add_argument("--context-tail-frac", type=float, default=None)
    ap.add_argument("--allow-recipe-mismatch", action="store_true",
                    help="score with prompt settings that contradict the model's "
                         "training config (ablations only; normally an error)")
    ap.add_argument("--logprobs", type=int, default=128,
                    help="logprob depth; deep enough that the losing answer token is "
                         "still returned, else saturated rows collapse to exactly 0/1 and "
                         "lose their ranking (at depth 20, 86%% of rows came out as 1.0)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="deterministic row cap for compatibility smoke tests; 0 = all")
    args = ap.parse_args()

    config_path, cfg = find_training_config(args.model, args.adapter)
    notes = resolve_prompt_recipe(args, cfg, config_path)
    if config_path is not None:
        print(f"[score] prompt recipe from {config_path}")
        for note in notes:
            print(f"[score]   {note}")
    else:
        print("[score] no training config found next to the model — using CLI/default "
              "prompt recipe (expected for untrained probe runs)")
    print(f"[score] recipe: " + " ".join(
        f"{k}={getattr(args, k)!r}" for k in RECIPE_DEFAULTS))

    data = Path(args.data_dir).expanduser()
    tdir = Path(args.transcripts_dir).expanduser()
    df = pd.read_csv(data / "train_features.csv")
    df["learning_objective"] = df["learning_objective"].str.strip()
    if args.fold >= 0:
        folds = pd.read_csv(args.folds)
        df = df.merge(folds, on="response_id", validate="1:1")
        df = df[df["fold"] == args.fold].reset_index(drop=True)
    # same-session rows adjacent -> shared transcript prefixes hit the vLLM prefix cache
    df = df.sort_values(["session_id", "response_id"], kind="stable").reset_index(drop=True)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    print(f"[score] {len(df):,} rows, model={args.model}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    yes_ids, no_ids = answer_token_ids(tok)

    print("[score] reading transcripts ...")
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
        print(f"[score] loaded objective context index: {len(context_index):,} rows")
    if args.prompt_mode == "objective_chunks" and args.require_context_index and not context_index:
        raise ValueError("objective_chunks with --require-context-index needs --context-index")
    if args.prompt_mode == "objective_chunks" and args.require_context_index:
        missing = set(df["response_id"].astype(str)) - set(context_index or {})
        if missing:
            sample = sorted(missing)[:3]
            raise ValueError(f"required context index is missing {len(missing):,} scored rows; sample={sample}")

    print("[score] building prompts ...")
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

    lora_kw, lora_req = {}, None
    if args.adapter:
        import json as _json
        from vllm.lora.request import LoRARequest
        r = _json.loads((Path(args.adapter).expanduser()
                         / "adapter_config.json").read_text())["r"]
        lora_kw = dict(enable_lora=True, max_lora_rank=max(r, 16))
        lora_req = LoRARequest("ft", 1, str(Path(args.adapter).expanduser()))
    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_input_tokens + 64,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_prefix_caching=True, max_logprobs=args.logprobs, **lora_kw)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=args.logprobs)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts],
                        sampling_params=sp, lora_request=lora_req)

    p_yes = np.full(len(df), np.nan)
    n_miss = 0
    for i, o in enumerate(outs):
        lp = o.outputs[0].logprobs[0]  # dict token_id -> Logprob
        py = max((np.exp(lp[t].logprob) for t in yes_ids if t in lp), default=0.0)
        pn = max((np.exp(lp[t].logprob) for t in no_ids if t in lp), default=0.0)
        if py + pn <= 0:
            n_miss += 1
            p_yes[i] = 0.5
        else:
            p_yes[i] = py / (py + pn)
    if n_miss:
        print(f"[score] WARNING: {n_miss} rows had no Yes/No in top-20 logprobs")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"response_id": df["response_id"], "p_llm": p_yes}).to_csv(out, index=False)
    print(f"[score] wrote {out}")

    labels = data / "train_labels.csv"
    if labels.exists():
        from sklearn.metrics import roc_auc_score
        from ace.utils import logloss
        m = pd.read_csv(labels).merge(
            pd.DataFrame({"response_id": df["response_id"], "p": p_yes}),
            on="response_id")
        print(f"[score] raw ll={logloss(m['is_correct'], m['p']):.4f} "
              f"auc={roc_auc_score(m['is_correct'], m['p']):.4f} "
              f"(raw = uncalibrated; Platt/blend under protocol v3 comes next)")


if __name__ == "__main__":
    main()
