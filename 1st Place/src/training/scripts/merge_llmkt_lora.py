"""Merge a dialogue-kt LoRA adapter into its base model and save plain merged weights.

The output directory holds standard bf16 ``safetensors`` shards plus the tokenizer, loadable by
``AutoModelForCausalLM.from_pretrained`` without peft. The submission ships the adapter and
merges it at inference: ``models/submission/main.py`` repeats this script's recipe (bf16 load
on CPU, ``merge_and_unload()``, then move to the device), because peft upcasts the ``B @ A``
product to fp32 only on CPU and a GPU merge gives slightly different bf16 weights.
Not run by ``reproduce.py``.

Usage (from build/workspace/, in the dialogue-kt environment, which has peft), e.g. for the
shipped adapter of step 06b:
    external/dialogue-kt/.env/bin/python scripts/merge_llmkt_lora.py \
        --adapter external/dialogue-kt/saved_models/lmkt_mrft_sa_8b_all35k \
        --base Qwen/Qwen3-8B --out artifacts/mrft_sa_8b_all35k_merged

Merging runs on CPU (no GPU needed) and takes ~15 GB RAM for a 7B model in bf16.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the training root (build/workspace/)
os.environ.setdefault("HF_HOME", f"{REPO}/external/dialogue-kt/.hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main():
    """Load the base in bf16, apply the adapter, merge it and save the result.

    Returns:
        None. Writes the merged weights (safetensors shards) and the base model's tokenizer to
        ``--out`` and prints the size of every file there.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="dir with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True, help="output dir for merged bf16 weights + tokenizer")
    ap.add_argument("--device", default="cpu", help="cpu (default, ~15GB RAM) or cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not os.path.exists(os.path.join(args.adapter, "adapter_config.json")):
        sys.exit(f"No adapter_config.json in {args.adapter}")

    print(f"loading base {args.base} (bf16) on {args.device} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    ).to(args.device)
    print(f"applying adapter {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter)
    print("merging ...", flush=True)
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    print(f"merged model + tokenizer -> {args.out}")
    # report shard sizes (no test content; just file sizes)
    total = 0
    for f in sorted(os.listdir(args.out)):
        p = os.path.join(args.out, f)
        sz = os.path.getsize(p)
        total += sz
        print(f"  {f}  ({sz:,} bytes)")
    print(f"total {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
