#!/usr/bin/env python
"""Standalone vLLM transcript-reader scorer for ONE LLM bundle — a subprocess worker.

main.py spawns one of these per LLM bundle (4B, 8B, ...) so each vLLM engine gets
a fully-isolated CUDA/process lifecycle: a fresh process that exits cleanly,
avoiding the flaky two-engine in-process teardown (the 8B smoke log's
destroy_process_group warning). Reads a rows CSV (response_id, session_id,
learning_objective), computes each row's transcript-reader logit
z_ft = mean logit over k adapters, writes (response_id, z_ft) to --out.

Invoked by main.py (python -m ace.llm.score_worker ...), not a user tool. The
vLLM env is set at module top so it applies before the vLLM import in this fresh
process. Competition logging rule: no counts/statistics derived from test data.

Engine construction is driven by the bundle's OPTIONAL `bundle.json`
(``ace.llm.bundle``). A bundle without one — every reader staged before E23 —
gets the legacy defaults and byte-identical engine kwargs, so the shipped 4B/8B
path is unchanged. A Qwen3.5 bundle instead declares
``language_model_only``/``adapter_mode`` and is pre-flighted against what vLLM
0.21 can actually serve. Nothing architecture-specific lives in this file.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="assets/llm_* dir (base/ + adapters/)")
    ap.add_argument("--rows", required=True, help="CSV: response_id, session_id, learning_objective")
    ap.add_argument("--transcripts", required=True, help="test_transcripts dir")
    ap.add_argument("--out", required=True, help="output CSV: response_id, z_ft")
    ap.add_argument("--serialize", type=int, default=1)
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--rowpass-budget", type=int, default=50_000)
    # Prompt recipe. Defaults reproduce the E13 full-transcript reader byte-for-byte,
    # so 4B/8B bundles are unaffected. A reader trained with objective_chunks MUST be
    # scored with the same recipe -- a mismatched prompt yields silently wrong
    # predictions (the footgun score_vllm.py's recipe guard exists to catch).
    ap.add_argument("--prompt-mode", default="standard",
                    choices=["standard", "objective_chunks"])
    ap.add_argument("--context-index", default=None,
                    help="JSONL semantic context index (objective_chunks only)")
    ap.add_argument("--context-full-threshold", type=int, default=0)
    ap.add_argument("--context-chunks", type=int, default=2)
    ap.add_argument("--context-head-frac", type=float, default=0.16)
    ap.add_argument("--context-tail-frac", type=float, default=0.34)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from ace.llm.bundle import (BASE_MODE_EMBEDDED, adapter_dirs, check_engine_supports,
                                engine_kwargs, load_bundle_meta, resolve_base_path,
                                validate_bundle)
    from ace.llm.objective_context import build_prompt_messages, load_context_index
    from ace.llm.serialize import answer_token_ids, apply_template, read_dialogue

    bundle = Path(args.bundle)
    tdir = Path(args.transcripts)
    df = pd.read_csv(args.rows)

    # Bundle metadata is OPTIONAL: a bundle with no bundle.json loads the legacy
    # defaults and every step below behaves exactly as it did before E23 (the 4B/8B
    # readers of SUB-7/SUB-9 are staged that way and are not regenerated).
    #
    # Resolve the base FIRST (embedded -> bundle/base; external -> the model the
    # runtime preloads), then validate whatever that resolution produced. One code path
    # for both modes is the point: an EXTERNAL Qwen3.5 checkpoint declaring the
    # text-only Qwen3_5ForCausalLM is rejected here with an explanation, rather than
    # skipped and left to fail as a vLLM registry traceback minutes into the load.
    meta = load_bundle_meta(bundle)
    base_path = resolve_base_path(bundle, meta, runtime_root=Path.cwd())
    validate_bundle(bundle, meta, runtime_root=Path.cwd())
    # Metadata only — no counts or statistics derived from the test data.
    print(f"[worker] bundle {bundle.name}: {meta.describe()}", flush=True)
    if meta.base_mode != BASE_MODE_EMBEDDED:
        print(f"[worker] base resolved from {meta.base_mode} source: {base_path}",
              flush=True)
    if meta.incomplete_cv:
        print(f"[worker] NOTE: incomplete CV — missing folds {meta.missing_folds}",
              flush=True)

    adapters = adapter_dirs(bundle)
    n_ad = len(adapters)
    # k adapts to test size via the row-pass budget (k=1 for a single full-fit
    # adapter). Same-session rows share adapters + stay adjacent for the prefix cache.
    # A dense bundle has no adapters: one pass, k=1.
    k = max(1, min(n_ad, int(args.rowpass_budget) // max(len(df), 1))) if n_ad else 1

    tok = AutoTokenizer.from_pretrained(str(base_path))
    yes_ids, no_ids = answer_token_ids(tok)

    sess = df["session_id"].to_numpy()
    order = np.argsort(sess, kind="stable")
    sids = sorted(df["session_id"].unique())
    workers = min(16, os.cpu_count() or 4)
    with Pool(workers) as pool:
        dialogues = dict(zip(sids, pool.map(
            partial(read_dialogue, version=int(args.serialize)),
            [str(tdir / f"{s}.csv") for s in sids], chunksize=32)))

    max_in = int(args.max_input_tokens)
    objectives = df["learning_objective"].to_numpy()
    rids = df["response_id"].to_numpy()

    context_index = None
    if args.context_index:
        context_index = load_context_index(args.context_index)
        bad = next((p for p in context_index.values() if p.serialize != int(args.serialize)),
                   None)
        if bad is not None:
            raise ValueError(f"context index serialize={bad.serialize} "
                             f"but --serialize={args.serialize}")
    if args.prompt_mode == "objective_chunks":
        # Required, not best-effort: falling back to the lexical plan for rows the
        # index missed would score part of the test set with a different prompt.
        if context_index is None:
            raise ValueError("--prompt-mode objective_chunks requires --context-index")
        missing = set(map(str, rids)) - set(context_index)
        if missing:
            raise ValueError(f"context index is missing {len(missing):,} scored rows")

    prompts = [apply_template(tok, build_prompt_messages(
                   tok, dialogues[s], o, max_input_tokens=max_in,
                   prompt_mode=args.prompt_mode, response_id=str(rid), session_id=str(s),
                   context_index=context_index,
                   require_context_index=args.prompt_mode == "objective_chunks",
                   context_full_threshold=args.context_full_threshold,
                   context_chunks=args.context_chunks,
                   context_head_frac=args.context_head_frac,
                   context_tail_frac=args.context_tail_frac))
               for rid, s, o in zip(rids[order], sess[order], objectives[order])]
    sess_code = pd.factorize(sess[order])[0]

    # torch.compile + CUDA graph capture cost ~40-120 s per engine (Phi-4: 74 s
    # compile + 46 s capture of its 158 s startup) and mostly pay off across many
    # decode steps -- but max_tokens=1 makes this workload prefill-dominated.
    # On a tiny input that startup dominates the wall clock: the 10-minute smoke
    # test ships 100 responses, and three sequential engines cannot afford it.
    # Drop to eager there; keep the compiled path for the full test set.
    # NB: deliberately not logged -- the rules forbid emitting any aggregate of
    # the test data, and the row count is one.
    # Capability check + runtime provenance. Blocks on ACTUAL incompatibility (no CUDA,
    # no vLLM, a build that cannot accept this bundle's engine arguments); a runtime
    # merely NEWER than the parity-certified reference only warns and continues.
    check_engine_supports(meta)
    kwargs = engine_kwargs(
        meta, model_path=base_path, max_input_tokens=max_in,
        enforce_eager=len(df) <= int(os.environ.get("ACE_EAGER_MAX_ROWS", "500")))
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=128)

    def _accumulate(idx, outs, z_sorted) -> None:
        """Yes/No -> logit, unchanged from the pre-E23 path (do not alter: the stack
        coefficients were fit against exactly this normalisation)."""
        for i, o in zip(idx, outs):
            lp = o.outputs[0].logprobs[0]
            py = max((np.exp(lp[t].logprob) for t in yes_ids if t in lp), default=0.0)
            pn = max((np.exp(lp[t].logprob) for t in no_ids if t in lp), default=0.0)
            p = py / (py + pn) if py + pn > 0 else 0.5
            p = min(max(p, 1e-6), 1 - 1e-6)
            z_sorted[i] += np.log(p / (1 - p))

    z_sorted = np.zeros(len(df))
    if n_ad == 0:
        # Dense bundle: the LoRA deltas are already merged into the checkpoint, so
        # there is one model and one pass over every row.
        all_idx = np.arange(len(df))
        outs = llm.generate([{"prompt_token_ids": prompts[i]} for i in all_idx],
                            sampling_params=sp, use_tqdm=False)
        _accumulate(all_idx, outs, z_sorted)
    else:
        for j, ad in enumerate(adapters):
            idx = np.flatnonzero(((j - sess_code) % n_ad) < k)
            if idx.size == 0:
                continue
            outs = llm.generate([{"prompt_token_ids": prompts[i]} for i in idx],
                                sampling_params=sp, use_tqdm=False,
                                lora_request=LoRARequest(ad.name, j + 1, str(ad)))
            _accumulate(idx, outs, z_sorted)
    z_sorted /= k
    z = np.empty(len(df))
    z[order] = z_sorted
    pd.DataFrame({"response_id": df["response_id"], "z_ft": z}).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()