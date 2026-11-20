import math
import os
import json
from tqdm import tqdm
import torch
import transformers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from pykt.models.dkt import DKT
from pykt.models.akt import AKT
from pykt.models.dkvmn import DKVMN
from pykt.models.saint import SAINT
from pyBKT.models import Model as BKT
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.lm import get_model
from dialogue_kt.models.dkt_multi_kc import DKTMultiKC
from dialogue_kt.models.dkt_sem import DKTSem
from dialogue_kt.models.simplekt import simpleKT
from dialogue_kt.data_loading import (load_annotated_data, load_monitor_val_data, get_kc_result_filename, get_qual_result_filename, get_default_fold, load_kc_dict,
                          correct_to_str, standards_to_str, get_model_file_suffix, COMTA_SUBJECTS)
from dialogue_kt.kt_data_loading import (LMKTDatasetUnpacked, LMKTCollatorUnpacked, LMKTDatasetPacked, LMKTCollatorPacked,
                             DKTDataset, DKTCollator, get_dataloader, apply_annotations)
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.utils import device, get_checkpoint_path

# ===== Common Functions =====

def apply_defaults(args):
    if args.model_type == "lmkt":
        defaults = {
            "epochs": 5,
            "lr": 2e-4,
            "wd": 1e-2,
            "gc": 1.0,
            "batch_size": 1,
            "grad_accum_steps": 64,
            "r": 16,
            "lora_alpha": 16
        }
    else:
        defaults = {
            "epochs": 100,
            "lr": 1e-3,
            "wd": 1e-2,
            "gc": 0,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "emb_size": 64
        }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    print("Arguments:", args)

def hyperparam_sweep(args):
    apply_defaults(args)
    args.testonval = True
    args.crossval = args.dataset == "comta"
    model_names = []
    results = []
    if args.model_type == "lmkt":
        for lr in [5e-5, 1e-4, 2e-4, 3e-4]:
            for r in [4, 8, 16, 32]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_lmkt_agg{args.agg}_lr{lr}_r{r}"
                args.lr = lr
                args.r = r
                args.lora_alpha = r
                model_names.append(args.model_name)
                results.append(train(args))
    else:
        for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
            for emb_size in [8, 16, 32, 64, 128, 256, 512]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}_lr{lr}_es{emb_size}"
                args.lr = lr
                args.emb_size = emb_size
                model_names.append(args.model_name)
                results.append(train(args))
    aucs = np.array([metrics.mean(0)[2] if args.crossval else metrics[2] for metrics in results])
    best_model_idx = aucs.argmax()
    result_str = "\n".join([f"{model_name}: {auc:.2f}" for model_name, auc in zip(model_names, aucs)])
    result_str += f"\nBest: {model_names[best_model_idx]}: {aucs[best_model_idx]:.2f}"
    print(result_str)
    with open(f"results/metrics_hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}.txt", "w") as file:
        file.write(result_str + "\n")

def crossval(args, fn):
    # Train/test models across folds
    metrics_agg = []
    folds = COMTA_SUBJECTS if args.split_by_subject else range(1, 6)
    for fold in folds:
        print(f"Fold {fold}...")
        metrics = fn(args, fold)
        metrics_agg.append(metrics)
    # Aggregate and report metrics across folds
    metrics_np = np.stack(metrics_agg, axis=0)
    avg = metrics_np.mean(axis=0)
    std = metrics_np.std(axis=0)
    metric_names = ["Loss", "Acc", "AUC", "Prec", "Rec", "F1"]
    if len(avg) > 6:
        metric_names += ["Acc (Final)", "AUC (Final)", "Prec (Final)", "Rec (Final)", "F1 (Final)"]
    results = [
        f"{metric}: ${avg[idx]:.2f}_{{\\pm {std[idx]:.2f}}}$" for idx, metric in
        enumerate(metric_names)
    ]
    result_str = "\n".join(results)
    print(result_str)
    with open(f"results/metrics_crossval{'_subj' if args.split_by_subject else ''}_{get_model_file_suffix(args)}.txt", "w") as out_file:
        out_file.writelines([
            str(metrics_agg) + "\n",
            result_str + "\n"
        ])
    # Aggregate and save qual analysis files across folds
    if args.model_type == "lmkt":
        dfs = [pd.read_csv(get_qual_result_filename(args, fold)) for fold in folds]
        pd.concat(dfs).to_csv(get_qual_result_filename(args), index=False)
    return metrics_np

def train(args):
    if args.hyperparam_sweep:
        args.hyperparam_sweep = False
        return hyperparam_sweep(args)

    assert args.model_name or args.model_type == "bkt"
    apply_defaults(args)
    fn = train_lmkt if args.model_type == "lmkt" else train_test_bkt if args.model_type == "bkt" else train_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def test(args):
    apply_defaults(args)
    fn = test_lmkt if args.model_type == "lmkt" else test_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def compute_metrics(labels, preds):
    hard_preds = np.round(preds)
    acc = accuracy_score(labels, hard_preds)
    auc = roc_auc_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard_preds, average="binary")
    return acc * 100, auc * 100, prec * 100, rec * 100, f1 * 100

def compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold):
    result_str = f"Loss: {loss:.4f}\n"
    result_str += f"Overall ({len(all_labels)} samples):\n"
    result_str += f"GT - True: {sum(all_labels)}, False: {len(all_labels) - sum(all_labels)}; "
    result_str += f"Pred - True: {sum(np.round(all_preds))}, False: {len(all_preds) - sum(np.round(all_preds))}\n"
    all_metrics = compute_metrics(all_labels, all_preds)
    result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*all_metrics)
    if final_turn_labels is not None:
        result_str += f"Final Turn ({len(final_turn_labels)} samples):\n"
        result_str += f"GT - True: {sum(final_turn_labels)}, False: {len(final_turn_labels) - sum(final_turn_labels)}; "
        result_str += f"Pred - True: {sum(np.round(final_turn_preds))}, False: {len(final_turn_preds) - sum(np.round(final_turn_preds))}\n"
        final_metrics = compute_metrics(final_turn_labels, final_turn_preds)
        result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*final_metrics)
    else:
        final_metrics = []
    print(result_str)
    with open(f"results/metrics_{get_model_file_suffix(args, fold)}.txt", "w") as out_file:
        out_file.write(result_str)
    return all_metrics, final_metrics


# ===== LMKT =====

def _agg_kc_probs_vectorized(kc_probs, num_kcs, agg):
    """Aggregate per-KC probabilities into per-turn correctness probabilities, vectorized.

    The flat per-KC probabilities are scattered into a padded [groups, max_kc] matrix and
    reduced over dim 1 (padding is 0 for a sum, 1 for a product), without the per-turn Python
    loop and its per-turn .tolist() sync. On CPU it is bit-identical to the loop for prod and
    mean-ar (the default) and within ~6e-8 for mean-geo. Used only with LLMKT_FAST_LOSS=1.

    Args:
        kc_probs: 1-D tensor of P(True) for every KC prompt, grouped by turn in order.
        num_kcs: 1-D tensor with the number of KCs of each turn.
        agg: "prod", "mean-ar" or "mean-geo".

    Returns:
        A 1-D tensor with one correctness probability per turn.
    """
    num_kcs = num_kcs.long()
    G = num_kcs.shape[0]
    max_kc = int(num_kcs.max())
    valid = torch.arange(max_kc, device=kc_probs.device).unsqueeze(0) < num_kcs.unsqueeze(1)
    if agg == "mean-ar":
        mat = torch.zeros(G, max_kc, dtype=kc_probs.dtype, device=kc_probs.device)
        mat[valid] = kc_probs  # row-major fill == group concatenation order
        return mat.sum(dim=1) / num_kcs.to(mat.dtype)
    mat = torch.ones(G, max_kc, dtype=kc_probs.dtype, device=kc_probs.device)
    mat[valid] = kc_probs
    prod = mat.prod(dim=1)
    return prod if agg == "prod" else prod ** (1.0 / num_kcs.to(prod.dtype))


def get_lmkt_loss_unpacked(model, batch, true_token, false_token, args, fast=False):
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
    if fast:
        # Fork change. Training fast path: vectorized aggregation without the per-turn Python
        # loop and .tolist() sync, and kc_probs_grouped (unused in training) is not built.
        # Inference keeps fast=False, so its loop and grouped-probabilities output are
        # unchanged.
        corr_probs = _agg_kc_probs_vectorized(kc_probs, batch["num_kcs"], args.agg)
        loss = torch.nn.BCELoss()(corr_probs.float(), batch["labels"].float())
        return loss, None, corr_probs
    # Get probability that all KCs are True for each turn in the batch
    num_kc_counter = 0
    kc_probs_grouped = []
    corr_probs = []
    for num_kcs in batch["num_kcs"]:
        kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
        if args.agg == "prod":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod()
        elif args.agg == "mean-ar":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].mean()
        elif args.agg == "mean-geo":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod() ** (1 / num_kcs)
        corr_probs.append(prob)
        num_kc_counter += num_kcs
    corr_probs = torch.stack(corr_probs)
    # Get BCE loss with correctness labels and predicted probabilities
    loss = torch.nn.BCELoss()(corr_probs.float(), batch["labels"].float())
    return loss, kc_probs_grouped, corr_probs

def get_lmkt_loss_packed(model, batch, true_token, false_token, args, fast=False):
    # Fork change. `fast` is accepted for the same signature as the unpacked loss (which has
    # a vectorized training path); the packed variant has no loop, so it is ignored here.
    # Invert attention mask.
    # `model.dtype` exists on a bare HF model but not on a DistributedDataParallel wrapper
    # (AttributeError: 'DistributedDataParallel' object has no attribute 'dtype'), so packed +
    # DDP needs this. Read through .module when wrapped, and fall back to a parameter's dtype
    # so this works for other wrappers too.
    inner = getattr(model, "module", model)
    model_dtype = getattr(inner, "dtype", None) or next(inner.parameters()).dtype
    attention_mask = batch["attention_mask"]
    min_dtype = torch.finfo(model_dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model_dtype)
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=attention_mask, position_ids=batch["position_ids"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
    kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
    # Get probability that all KCs are True for each turn in the batch
    kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])]
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, batch["last_idxs"].to(device) == 0, torch.full_like(kc_probs, padding_val).to(device))
    # Get BCE loss with correctness labels and predicted probabilities
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=1)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=1) / batch["num_kcs"]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=1) ** (1 / batch["num_kcs"])
    loss = torch.nn.BCELoss()(corr_probs.float(), batch["labels"].float())
    return loss, kc_probs_grouped, corr_probs

STEP_LOG_HEADER = "run,epoch,opt_step,split,loss,n"

def _init_step_log(path):
    """Prepare the step-level loss log and return this launch's run id.

    A restart appends to the file the failed attempt was writing, so without a run tag the
    curves of different attempts run together. Tagging every row with a run id keeps a resumed
    run plottable. An old-format "epoch,step,loss" log is migrated in place to the current
    schema (its rows become run 0, split=train).

    Args:
        path: log file, normally saved_models/<name>/train_log.csv.

    Returns:
        The run id for this launch: 0 for a new file, else one more than the largest id in it
        (1 after a migration). Creates the file with its header if it does not exist.
    """
    import os
    run = 0
    if os.path.exists(path):
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        if lines and lines[0] != STEP_LOG_HEADER:
            migrated = [STEP_LOG_HEADER] + [
                f"0,{p[0]},{p[1]},train,{p[2]}," for p in (ln.split(",") for ln in lines[1:]) if len(p) == 3
            ]
            with open(path, "w") as fh:
                fh.write("\n".join(migrated) + "\n")
            run = 1
        else:
            runs = [int(ln.split(",")[0]) for ln in lines[1:]]
            run = max(runs) + 1 if runs else 0
    else:
        with open(path, "w") as fh:
            fh.write(STEP_LOG_HEADER + "\n")
    return run

def _monitor_val_loss(model, loader, get_loss, true_token, false_token, args, fast, max_batches, use_workers):
    """Compute the mean loss over a monitoring loader, without gradients.

    Args:
        model: the (unwrapped) LLMKT model.
        loader: DataLoader of the monitor set.
        get_loss: the loss function in use (packed or unpacked).
        true_token: token id of "True".
        false_token: token id of "False".
        args: parsed command-line arguments.
        fast: passed to ``get_loss`` (vectorized aggregation).
        max_batches: stop after this many batches (0 = all).
        use_workers: True if batches arrive on the CPU and must be copied to the GPU.

    Returns:
        A tuple ``(mean_loss, n_batches)``; the mean is NaN if no batch was scored. The model's
        train/eval mode is restored on exit.
    """
    was_training = model.training
    model.eval()
    total, n_batches = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if use_workers:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, _, _ = get_loss(model, batch, true_token, false_token, args, fast=fast)
            total += loss.item()
            n_batches += 1
    if was_training:
        model.train()
    return (total / n_batches if n_batches else float("nan")), n_batches

def train_lmkt(args, fold):
    # Fork change. Multi-GPU data parallelism, launched with torchrun (WORLD_SIZE > 1).
    # Per-rank grad_accum is divided by the world size, so the
    # global batch and the number of optimizer steps per epoch match the single-GPU run.
    import os
    import contextlib
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank0 = (not ddp) or local_rank == 0
    if ddp:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        args.grad_accum_steps = max(1, args.grad_accum_steps // world_size)
        if rank0:
            print(f"[ddp] world_size={world_size}, per-rank grad_accum={args.grad_accum_steps}")

    # Load language model with trainable LoRA adapters
    model, tokenizer = get_model(args.base_model, False, pt_model_name=args.pt_model_name, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize)
    if rank0:
        model.print_trainable_parameters()
    unwrapped_model = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Load and split dataset, annotated with correctness and KCs
    KTDataset = LMKTDatasetPacked if args.pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if args.pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if args.pack_kcs else get_lmkt_loss_unpacked
    train_df, val_df, _ = load_annotated_data(args, fold)
    if getattr(args, "max_train", None) and len(train_df) > args.max_train:
        train_df = train_df.sample(n=args.max_train, random_state=221)
        val_cap = max(1, args.max_train // 8)
        if len(val_df) > val_cap:
            val_df = val_df.sample(n=val_cap, random_state=221)
        print(f"[max_train] subsampled train to {len(train_df)}, val to {len(val_df)}")
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    train_dataset = KTDataset(train_df, tokenizer, args)
    val_dataset = KTDataset(val_df, tokenizer, args)
    # Pipelined data loading: LLMKT_DATALOADER_WORKERS>0
    # collates in worker processes. It needs a CPU-tensor collator (to_device=False) and the
    # host-to-device copy in the loop below. Unpacked path only; the default (0) keeps the
    # on-device collator.
    n_workers = int(os.environ.get("LLMKT_DATALOADER_WORKERS", "0"))
    use_workers = n_workers > 0 and not args.pack_kcs
    _pin = use_workers
    # Vectorized fast loss (LLMKT_FAST_LOSS=1, unpacked only): skips the per-turn Python loop
    # and .tolist() sync in get_lmkt_loss_unpacked.
    # Inference does not use it.
    _fast_loss = os.environ.get("LLMKT_FAST_LOSS") == "1" and not args.pack_kcs
    if _fast_loss and rank0:
        print(f"[loss] vectorized fast path ON (agg={args.agg}; per-turn loop/.tolist() skipped)")
    collator = KTCollator(tokenizer, to_device=False) if use_workers else KTCollator(tokenizer)
    if use_workers and rank0:
        print(f"[dataloader] pipelined: {n_workers} workers/rank, pin_memory, device-copy in loop")
    if ddp:
        # Length-grouped (bucketed) batching (LLMKT_GROUP_BY_LENGTH=1; optional): each DDP
        # step's batches have similar lengths on all ranks. The global batch and number of
        # optimizer steps are those of the default sampler; only the order changes.
        if os.environ.get("LLMKT_GROUP_BY_LENGTH") == "1" and not args.pack_kcs and hasattr(train_dataset, "lengths"):
            from dialogue_kt.kt_data_loading import DistributedLengthGroupedSampler
            train_sampler = DistributedLengthGroupedSampler(
                train_dataset.lengths, args.batch_size, dist.get_world_size(), dist.get_rank(), seed=221)
            if rank0:
                print(f"[ddp] length-grouped sampler ON: bucketed batches over {len(train_dataset.lengths)} samples")
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=221)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if len(val_dataset) else None
        train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, False, sampler=train_sampler, num_workers=n_workers if use_workers else 0, pin_memory=_pin)
        val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False, sampler=val_sampler, num_workers=n_workers if use_workers else 0, pin_memory=_pin)
    else:
        train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True, num_workers=n_workers if use_workers else 0, pin_memory=_pin)
        val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False, num_workers=n_workers if use_workers else 0, pin_memory=_pin)

    # Monitoring-only validation curve (LLMKT_VAL_PKL).
    # Built on rank 0 only and scored with the unwrapped model, so it never touches
    # DDP's reducer. It does not select checkpoints: under --train_all the last epoch is still
    # saved; this only adds rows to train_log.csv.
    monitor_loader = None
    monitor_val_df = load_monitor_val_data(args, fold) if rank0 else None
    if monitor_val_df is not None:
        if len(monitor_val_df) == 0:
            print("[val-monitor] LLMKT_VAL_PKL selected 0 rows; no validation curve will be logged.")
        else:
            has_lo = "lo_id" in train_df.columns and "lo_id" in monitor_val_df.columns
            shared = (set(train_df["lo_id"].astype(str)) & set(monitor_val_df["lo_id"].astype(str))) if has_lo else set()
            if shared:
                print(f"[val-monitor] WARNING: {len(shared)} learning objectives are in both train and the "
                      f"monitor set, so this curve is an optimistic bound, not the LO-disjoint view.")
            monitor_loader = get_dataloader(KTDataset(monitor_val_df, tokenizer, args), collator, args.batch_size,
                                            False, num_workers=n_workers if use_workers else 0, pin_memory=_pin)
            print(f"[val-monitor] {len(monitor_val_df)} samples, "
                  f"LO-disjoint={not shared if has_lo else 'unknown (no lo_id)'}, "
                  f"monitor-only (never used for checkpoint selection)")

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Do training loop
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.wd, relative_step=False)

    # Fork change. LR schedule. The defaults (schedule="constant", warmup_ratio=0) give a flat
    # lr for every step, the plain AdamW behaviour of upstream, which the pipeline uses.
    #
    # The step count must match the loop's boundary rule at `is_boundary` below:
    # ceil(len(dataloader) / grad_accum_steps) per epoch. `args.grad_accum_steps` was already
    # divided by world_size at the top of this function, and len(train_dataloader) is per rank
    # under a DistributedSampler, so this is the per-rank optimizer-step count, which is the
    # count the scheduler is stepped at. Do not divide it by world_size again.
    _steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    _total_steps = _steps_per_epoch * args.epochs
    _warmup_steps = int(round(getattr(args, "warmup_ratio", 0.0) * _total_steps))
    lr_scheduler = transformers.get_scheduler(
        getattr(args, "schedule", "constant"), optimizer,
        num_warmup_steps=_warmup_steps, num_training_steps=_total_steps)
    if rank0:
        # Printed so that a wrongly sized or inactive schedule shows in the log. A run that
        # reports warmup=1 is reading a dataset far smaller than intended.
        print(f"[sched] {getattr(args, 'schedule', 'constant')} | peak_lr={args.lr} | "
              f"steps/epoch={_steps_per_epoch} x {args.epochs} epochs = {_total_steps} total | "
              f"warmup={_warmup_steps} ({getattr(args, 'warmup_ratio', 0.0):.1%})", flush=True)
    best_val_loss = None
    # Mid-epoch checkpointing: SAVE_EVERY_STEPS>0
    # saves the adapter every N optimizer steps to `<ckpt>_midepoch`, not the main checkpoint
    # directory (see the save below), so this unconditional write cannot replace the
    # best-validation weights.
    _save_every = int(os.environ.get("SAVE_EVERY_STEPS", "0"))
    # Validation curve cadence: score the monitor set every N optimizer steps
    # (VAL_MAX_BATCHES caps its cost). 0 = at epoch end only (default).
    _val_every = int(os.environ.get("VAL_EVERY_STEPS", "0"))
    _val_max_batches = int(os.environ.get("VAL_MAX_BATCHES", "0"))
    # Step-level loss log (rank 0): saved_models/<name>/train_log.csv, one row per optimizer
    # step with the mean micro-batch loss since the previous step, plus a split=val row
    # wherever the monitor set was scored. Opened once per launch, so the run id is the same
    # across epochs.
    step_log_path, run_id = None, 0
    if rank0:
        ckpt_dir = get_checkpoint_path(args.model_name + (f"_{fold}" if fold else ""))
        os.makedirs(ckpt_dir, exist_ok=True)
        step_log_path = os.path.join(ckpt_dir, "train_log.csv")
        run_id = _init_step_log(step_log_path)
        if run_id:
            print(f"[step-log] appending as run {run_id} (train_log.csv already held {run_id} earlier run(s))")

    def _log_val(epoch, opt_step):
        """Score the monitor set, append a split=val row to the step log and print it.

        Rank 0 only; does nothing when no monitor set is configured.

        Args:
            epoch: 0-based epoch index.
            opt_step: optimizer steps done in this epoch.
        """
        if monitor_loader is None:
            return
        import time
        t0 = time.time()
        vl, nb = _monitor_val_loss(unwrapped_model, monitor_loader, get_loss, true_token, false_token,
                                   args, _fast_loss, _val_max_batches, use_workers)
        dt = time.time() - t0
        with open(step_log_path, "a") as fh:
            fh.write(f"{run_id},{epoch},{opt_step},val,{vl:.5f},{nb}\n")
        # Print the cost too: VAL_EVERY_STEPS adds wall-clock time and this is the only place
        # it shows. Overhead = dt / (VAL_EVERY_STEPS * seconds per optimizer step).
        print(f"[val-monitor] epoch {epoch} opt_step {opt_step}: val loss {vl:.4f} "
              f"({nb} batches, {dt:.1f}s)", flush=True)

    for epoch in range(args.epochs):
        if rank0:
            print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        if ddp:
            train_sampler.set_epoch(epoch)
        model.train()
        window_loss, opt_step = None, 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training", disable=not rank0)):
            if use_workers:  # worker-collated batches arrive on CPU (pinned); copy to GPU here
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            is_boundary = (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1
            # DDP gradient accumulation: allreduce gradients only on the optimizer-step boundary
            # (no_sync on the micro-batches in between). The result is the same as syncing every
            # micro-batch, because summing over micro-batches and averaging across ranks
            # commute. Every rank gets the same number of micro-batches (both
            # DistributedSampler and our grouped sampler guarantee it), so all ranks reach the
            # boundary together and cannot deadlock.
            sync_ctx = model.no_sync() if (ddp and not is_boundary) else contextlib.nullcontext()
            with sync_ctx:
                loss, _, _ = get_loss(model, batch, true_token, false_token, args, fast=_fast_loss)
                (loss / args.grad_accum_steps).backward()
            # Accumulate the loss on the GPU (detached) so there is no CUDA sync every
            # micro-batch; .item() is called only at the optimizer-step boundary (step log) and
            # at epoch end (average).
            _ld = loss.detach()
            total_train_loss = total_train_loss + _ld
            window_loss = _ld if window_loss is None else window_loss + _ld
            if is_boundary:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                # The scheduler steps here, on the optimizer-step boundary, not per micro-batch.
                # Stepping per micro-batch would advance the schedule grad_accum_steps times too
                # fast and leave the run at lr 0 for most of training.
                lr_scheduler.step()
                optimizer.zero_grad()
                opt_step += 1
                if step_log_path:
                    n = (batch_idx % args.grad_accum_steps) + 1
                    with open(step_log_path, "a") as fh:
                        fh.write(f"{run_id},{epoch},{opt_step},train,{window_loss.item() / n:.5f},{n}\n")
                window_loss = None
                if _save_every and rank0 and opt_step % _save_every == 0:
                    # A separate directory: this save is unconditional, while the epoch-end save
                    # at the bottom of the loop depends on the validation loss improving.
                    unwrapped_model.save_pretrained(
                        get_checkpoint_path(args.model_name + (f"_{fold}" if fold else "")) + "_midepoch")
                    print(f"[mid-epoch ckpt] saved adapter at opt_step {opt_step}", flush=True)
                if _val_every and opt_step % _val_every == 0:
                    if rank0:
                        _log_val(epoch, opt_step)
                    if ddp:
                        dist.barrier()

        avg_train_loss = float(total_train_loss) / len(train_dataloader)  # float() = single epoch-end sync
        if ddp:  # average train loss across ranks for reporting
            t = torch.tensor([avg_train_loss], device=f"cuda:{local_rank}")
            dist.all_reduce(t)
            avg_train_loss = (t / world_size).item()
        model_name = args.model_name + (f"_{fold}" if fold else "")
        if rank0 and step_log_path:  # epoch mean, so runs with different step counts stay comparable
            with open(step_log_path, "a") as fh:
                fh.write(f"{run_id},{epoch},{opt_step},train_epoch,{avg_train_loss:.5f},{len(train_dataloader)}\n")
        if len(val_dataloader) == 0:
            # --train_all: no held-out val to select on; save every epoch (last wins).
            if rank0:
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: n/a (train_all)")
                _log_val(epoch, opt_step)  # monitor set only; still does not select
                print("Saving model (train_all, no val selection)...")
                unwrapped_model.save_pretrained(get_checkpoint_path(model_name))
            if ddp:
                dist.barrier()
        else:
            with torch.no_grad():
                model.eval()
                for batch in tqdm(val_dataloader, desc="Validating", disable=not rank0):
                    if use_workers:
                        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    loss, _, _ = get_loss(model, batch, true_token, false_token, args, fast=_fast_loss)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            if ddp:  # DistributedSampler pads val to a multiple of world_size; the few
                     # duplicated samples bias all epochs identically, so selection is fair.
                t = torch.tensor([avg_val_loss], device=f"cuda:{local_rank}")
                dist.all_reduce(t)
                avg_val_loss = (t / world_size).item()
            if rank0:
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                if step_log_path:
                    with open(step_log_path, "a") as fh:
                        fh.write(f"{run_id},{epoch},{opt_step},val,{avg_val_loss:.5f},{len(val_dataloader)}\n")
            if rank0:
                ep_path = get_checkpoint_path(model_name) + f"_epoch{epoch + 1}"
                unwrapped_model.save_pretrained(ep_path)
                print(f"[epoch ckpt] saved {ep_path}", flush=True)
            if not best_val_loss or avg_val_loss < best_val_loss:
                if rank0:
                    print("Best! Saving model...")
                    unwrapped_model.save_pretrained(get_checkpoint_path(model_name))
                best_val_loss = avg_val_loss
            if ddp:
                dist.barrier()

    if getattr(args, "train_all", False):
        if rank0:
            print("[train_all] training complete; skipping held-out test phase.")
        return None
    if ddp:
        # Only rank 0 runs the held-out test; free training memory first. Other ranks
        # exit cleanly (torchrun tolerates clean exits while rank 0 continues).
        dist.barrier()
        dist.destroy_process_group()
        if not rank0:
            return None
        del model, unwrapped_model, optimizer
        torch.cuda.empty_cache()
    return test_lmkt(args, fold)

def test_lmkt(args, fold):
    # Load trained language model
    model_name = args.model_name and args.model_name + (f"_{fold}" if fold else "")
    model, tokenizer = get_model(args.base_model, True, model_name=model_name, quantize=args.quantize)
    model.eval()

    # Load annotated data
    KTDataset = LMKTDatasetPacked if args.pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if args.pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if args.pack_kcs else get_lmkt_loss_unpacked
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if getattr(args, "max_test", None) and len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=221)
        print(f"[max_test] subsampled test set to {len(test_df)}")
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    # Fork change. Optional evaluation shard: LLMKT_TEST_SHARD="i/N" keeps every Nth test
    # dialogue, so one fold's test set can be split across N GPUs and the qual/kcs outputs
    # concatenated. Rows carry `Dialogue ID`, so the shards can be joined. The default "0/1"
    # changes nothing.
    _shard = os.environ.get("LLMKT_TEST_SHARD", "0/1")
    _si, _sn = (int(x) for x in _shard.split("/"))
    if _sn > 1:
        test_df = test_df.iloc[_si::_sn]
        print(f"TEST SHARD {_si}/{_sn}: {len(test_df)} dialogues")
    test_dataset = KTDataset(test_df, tokenizer, args, skip_first_turn=not args.inc_first_label)
    collator = KTCollator(tokenizer)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Collect meta data and predicted KC/correctness probabilities for test set
    dialogue_idx_to_sample_idxs = {}
    all_labels = []
    all_preds = []
    all_kc_probs = []
    all_kcs = []
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        for sample_idx, sample in enumerate(batch["meta_data"]):
            dialogue_idx_to_sample_idxs.setdefault(sample["dialogue_idx"], []).append(len(all_labels) + sample_idx)
        with torch.no_grad():
            loss, kc_probs, corr_probs = get_loss(model, batch, true_token, false_token, args)
        total_loss += loss.item()
        all_labels.extend(batch["labels"].tolist())
        all_preds.extend(corr_probs.tolist())
        all_kc_probs.extend(kc_probs)
        all_kcs.extend([sample["kcs"] for sample in batch["meta_data"]])

    # Compute quantitative metrics and save metrics file
    loss = total_loss / len(test_dataloader)
    final_turn_labels = [all_labels[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    final_turn_preds = [all_preds[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    all_metrics, final_metrics = compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold)

    # Save file for visual analysis
    kc_results = {
        dialogue_idx: [
            {
                kc: kc_prob
                for kc, kc_prob in zip(all_kcs[sample_idx], all_kc_probs[sample_idx])
            }
            for sample_idx in sample_idxs
        ]
        for dialogue_idx, sample_idxs in dialogue_idx_to_sample_idxs.items()
    }
    with open(get_kc_result_filename(args, fold), "w") as out_file:
        json.dump(kc_results, out_file, indent=2)

    # Save file for qualitative analysis
    qual_data = []
    for dia_idx, sample in test_df.iterrows():
        dialogue = apply_annotations(sample)
        if dia_idx not in dialogue_idx_to_sample_idxs:
            continue
        dia_preds = [all_preds[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_labels = [all_labels[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_acc = f"{(np.round(dia_preds) == dia_labels).mean():.4f}"
        first_turn = not args.inc_first_label
        label_counter = 0
        for turn in dialogue:
            if turn["correct"] is not None and not first_turn:
                label_idx = dialogue_idx_to_sample_idxs[dia_idx][label_counter]
                prob = f"{all_preds[label_idx]:.4f}"
                kc_probs = ", ".join([f"{kc_prob:.4f}" for kc_prob in all_kc_probs[label_idx]])
                label_counter += 1
            else:
                prob = "--"
                kc_probs = "--"
            if turn["correct"] is not None and first_turn:
                first_turn = False
            qual_data.append({
                "Dialogue ID": dia_idx,
                "Turn": turn["turn"],
                "Teacher": turn["teacher"] or "--",
                "Student": turn["student"],
                "Correct": correct_to_str(turn["correct"]),
                "Prob": prob,
                "KC Probs": kc_probs,
                "Dialogue Acc.": dia_acc,
                "KCs": standards_to_str(turn["kcs"], "\n"),
                "Notes": ""
            })
        qual_data.append({key: "" for key in qual_data[0]})
    pd.DataFrame(qual_data).to_csv(get_qual_result_filename(args, fold), index=False)

    return np.array([loss, *all_metrics, *final_metrics])


# ===== Baselines =====

BASELINE_MODELS = ["dkt-multi", "dkt-sem", "dkt", "akt", "dkvmn", "saint", "simplekt"]
NON_FLAT_KC_ARCH = ["dkt-multi", "dkt-sem"]

def select_flat_baseline_out_vectors(y: torch.Tensor, batch, shift_turn_end_idxs: bool):
    if shift_turn_end_idxs:
        # Predict KCs with output from first KC of turn for models where correctness is only visible in previous idxs
        # Clip at end to prevent out of bounds, no effect since last pred unused
        batch["turn_end_idxs"] = torch.clip(batch["turn_end_idxs"] + 1, max=batch["turn_end_idxs"].max())
    # Get output vectors at index of last KC per turn (to predict next turn's KCs)
    turn_end_idxs = batch["turn_end_idxs"].unsqueeze(2).repeat(1, 1, y.shape[2])
    return torch.gather(y, 1, turn_end_idxs)

def get_baseline_loss(y: torch.Tensor, batch, args):
    # Aggregate KC probs from outputs, one output per question
    batch_size, max_seq_len, max_num_kcs = batch["kc_ids"].shape
    kc_pad_mask = torch.arange(max_num_kcs).repeat(batch_size, max_seq_len, 1).to(device) >= batch["num_kcs"].unsqueeze(2)
    y = y[:, :-1].contiguous() # Last item in sequence doesn't predict anything
    kc_probs = torch.gather(y, 2, batch["kc_ids"][:, 1:]) # Collect KC predictions for next question, B x L x K
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, kc_pad_mask[:, 1:], torch.full_like(kc_probs, padding_val).to(device))
    # Calculate correct probabilities (B x L)
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=2)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=2) / batch["num_kcs"][:, 1:]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=2) ** (1 / batch["num_kcs"][:, 1:])

    # Compute BCE loss
    labels_flat = batch["labels"][:, 1:].contiguous().view(-1)
    loss_mask = labels_flat != -100
    labels_flat = labels_flat[loss_mask].type(torch.float)
    corr_probs_flat = corr_probs.view(-1)[loss_mask]
    loss: torch.Tensor = torch.nn.BCELoss()(corr_probs_flat, labels_flat)
    return loss, corr_probs

def get_baseline_model(kc_dict: dict, kc_emb_matrix: torch.Tensor, args):
    num_kcs = len(kc_dict)
    emb_size = args.emb_size
    n_blocks = 4 # For layered models
    if args.model_type == "dkt-multi":
        return DKTMultiKC(num_kcs, emb_size).to(device)
    if args.model_type == "dkt-sem":
        return DKTSem(emb_size, kc_emb_matrix).to(device)
    if args.model_type == "dkt":
        return DKT(num_kcs, emb_size).to(device)
    if args.model_type == "akt":
        model = AKT(num_kcs, num_kcs, emb_size, n_blocks, 0.05, emb_size, final_fc_dim=emb_size)
        model.out[3] = torch.nn.Linear(emb_size, emb_size) # Reduce from 256 to emb_size to avoid overparameterization
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "dkvmn":
        model = DKVMN(num_kcs, emb_size, 50)
        model.p_layer = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "saint":
        model = SAINT(num_kcs, num_kcs, 256, emb_size, 8, 0.2, n_blocks)
        model.out = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "simplekt":
        model = simpleKT(num_kcs, num_kcs, emb_size, n_blocks, 0.2, d_ff=emb_size, final_fc_dim=emb_size, final_fc_dim2=emb_size)
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    raise Exception(f"Model {args.model_type} not supported")

def compute_baseline_loss(model, batch, args):
    if args.model_type == "dkt-multi":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt-sem":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, False)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "akt":
        y, rasch_loss = model(batch["kc_ids_flat"], batch["labels_flat"], batch["kc_ids_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        loss, corr_probs = get_baseline_loss(y, batch, args)
        loss += rasch_loss
        return loss, corr_probs
    elif args.model_type == "dkvmn":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "saint":
        y = model(batch["kc_ids_flat"], batch["kc_ids_flat"], batch["labels_flat"][:, :-1])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "simplekt":
        y = model({
            "qseqs": batch["kc_ids_flat"][:, :-1],
            "cseqs": batch["kc_ids_flat"][:, :-1],
            "rseqs": batch["labels_flat"][:, :-1],
            "shft_qseqs": batch["kc_ids_flat"][:, 1:],
            "shft_cseqs": batch["kc_ids_flat"][:, 1:],
            "shft_rseqs": batch["labels_flat"][:, 1:]
        })
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    raise Exception(f"Model {args.model_type} not supported")

def compute_kc_emb_matrix(sbert_model: SentenceTransformer, kc_dict: dict):
    print("Computing SBERT embeddings...")
    kcs = [kv[0] for kv in sorted(kc_dict.items(), key=lambda kv: kv[1])]
    kc_emb_matrix = sbert_model.encode(kcs, convert_to_tensor=True)
    return kc_emb_matrix

def train_baseline(args, fold):
    assert args.model_type in BASELINE_MODELS

    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Create model
    model = get_baseline_model(kc_dict, kc_emb_matrix, args)

    # Load and split dataset, annotated with correctness and KCs
    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    train_dataset = DKTDataset(train_df, kc_dict, kc_emb_matrix, sbert_model)
    val_dataset = DKTDataset(val_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # Do training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _ = compute_baseline_loss(model, batch, args)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _ = compute_baseline_loss(model, batch, args)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
            torch.save(model.state_dict(), get_checkpoint_path(model_name))
            best_val_loss = avg_val_loss

    return test_baseline(args, fold)

def test_baseline(args, fold):
    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Load trained model
    if args.model_type in BASELINE_MODELS:
        model = get_baseline_model(kc_dict, kc_emb_matrix, args)
        model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
        model.load_state_dict(torch.load(get_checkpoint_path(model_name), map_location=device))
        model.eval()
    else:
        model = None

    # Load annotated data
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    test_dataset = DKTDataset(test_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # Collect meta data and predicted KC/correctness probabilities for test set
    all_labels = []
    all_preds = []
    final_turn_labels = []
    final_turn_preds = []
    total_loss = 0
    for batch in tqdm(test_dataloader):
        labels = batch["labels"][:, 1:]
        if model is not None:
            with torch.no_grad():
                loss, corr_probs = compute_baseline_loss(model, batch, args)
        elif args.model_type == "random":
            corr_probs = torch.zeros_like(labels).random_(0, 2)
            loss = torch.tensor(0)
        elif args.model_type == "majority":
            corr_probs = torch.full_like(labels, fill_value=test_dataset.majority_class)
            loss = torch.tensor(0)
        total_loss += loss.item()
        mask = labels != -100
        all_labels.extend(labels[mask].tolist())
        all_preds.extend(corr_probs[mask].tolist())
        final_idxs = mask.sum(dim=1) - 1
        final_turn_labels.extend(labels[torch.arange(mask.shape[0]), final_idxs].tolist())
        final_turn_preds.extend(corr_probs[torch.arange(mask.shape[0]), final_idxs].tolist())

    # Compute quantitative metrics across all turns and only on final turns
    loss = total_loss / len(test_dataloader)
    all_metrics, final_metrics = compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold)

    return np.array([loss, *all_metrics, *final_metrics])

def bkt_prep_data(df: pd.DataFrame, kc_dict: dict):
    dataset = DKTDataset(df, kc_dict, None, None)
    results = []
    order_id = 0
    for _, sample in enumerate(dataset.data):
        for kc, label in zip(sample["kc_ids_flat"], sample["labels_flat"]):
            results.append({"user_id": sample["dialogue_idx"], "skill_name": str(kc), "correct": label, "order_id": order_id})
            order_id += 1
    return pd.DataFrame(results), dataset

def train_test_bkt(args, fold):
    # Load and reformat data
    kc_dict = load_kc_dict(args)
    train_df, val_df, test_df = load_annotated_data(args, fold)
    train_df, _ = bkt_prep_data(pd.concat([train_df, val_df]), kc_dict)
    test_df, test_dataset = bkt_prep_data(test_df, kc_dict)

    # Train model
    model = BKT(seed=221, num_fits=1)
    model.fit(data=train_df)

    # Test model
    print("Train Acc./AUC:", model.evaluate(data=train_df, metric=["accuracy", "auc"]))
    print("Test Acc./AUC:", model.evaluate(data=test_df, metric=["accuracy", "auc"]))
    pred_df: pd.DataFrame = model.predict(data=test_df)
    pred_df = pred_df.sort_values(["order_id"])
    all_labels = []
    all_preds = []
    for sample, (_, user) in zip(test_dataset, pred_df.groupby("user_id", sort=False)):
        preds_flat = user["correct_predictions"]
        labels = sample["labels"]
        preds = []
        prev_idx = 0
        for turn_end_idx in sample["turn_end_idxs"]:
            if args.agg == "prod":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-ar":
                preds.append(np.mean(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-geo":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]) ** (1 / (turn_end_idx - prev_idx + 1)))
            prev_idx = turn_end_idx + 1
        all_labels.extend(labels[1:])
        all_preds.extend(preds[1:])
    metrics, _ = compute_all_metrics(0, all_labels, all_preds, None, None, args, fold)
    return [0, *metrics]
