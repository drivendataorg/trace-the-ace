import os
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.dkt_sem import ALT_ARCH
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, dkt_sem_prompt
from dialogue_kt.utils import device

def apply_annotations(sample: dict, apply_na: bool = True):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    # Handle dialogues beginning with turn 0 (student-initiated)
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    # Copy correctness and kcs into dialogue
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr = anno_turn["correct"]
        kcs = anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    # Use human annotation of correctness for final turn
    if dialogue[-1]["kcs"]: # Skip if no KCs for final turn since correct must be None
        if "expected_result" in sample["meta_data"]: # CoMTA
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"]: # MathDial
            if dialogue[-1]["correct"] is not None: # Final turn could be closing remarks, so skip if not tagged as having correctness
                if sample["meta_data"]["self_correctness"] == "Yes":
                    dialogue[-1]["correct"] = True
                elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                    dialogue[-1]["correct"] = None
                elif sample["meta_data"]["self_correctness"] == "No":
                    dialogue[-1]["correct"] = False
    return dialogue

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class LMKTDatasetUnpacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        # Fork change. End at the model's own generation prompt, so the next
                        # predicted token is True/False for any chat model (Llama, Qwen, ...).
                        # For Llama this is identical to the original (the prediction token is the
                        # same '\n\n'); the original's hardcoded assistant-"\n" turn and sum-2
                        # offset only worked with Llama's header format. See LMKTCollatorUnpacked.
                        # enable_thinking=False: Qwen3's template then inserts an empty
                        # <think></think> block after the generation prompt, so the first
                        # generated token is the answer (True/False), not a reasoning token, which
                        # is what the last-real-token logit read needs. Templates without a
                        # thinking mode (Qwen2.5, Llama) ignore the argument.
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], kc, args)},
                        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")
        # Per-sample length proxy (character length of the longest prompt) for length-grouped
        # DDP batching (DistributedLengthGroupedSampler below).
        self.lengths = [max((len(p) for p in s["prompts"]), default=1) for s in self.data]

# Fork addition. Cap the tokenized prompt length to bound bf16 activation and logit memory
# (the full-vocabulary logits tensor [seq_len x 152k] spikes on long dialogues and runs out
# of memory even on an 80GB A100). Prompts are left-truncated (the most recent tokens are
# kept), so the terminal turn, the objective and the True/False generation-prompt position,
# all at the end, are preserved; only the oldest turns of a few very long dialogues are
# dropped, which matches the end-of-session window the task is about. The default is 4096;
# with --select full the transcript runs to ~8.4k tokens, so the pipeline sets
# LLMKT_MAX_SEQ_LEN=8192 (H100). It is an environment variable so that training (here) and
# inference (models/submission/assets/magnificat/llmkt_infer.py) use the same value.
MAX_SEQ_LEN = int(os.environ.get("LLMKT_MAX_SEQ_LEN", "4096"))

class LMKTCollatorUnpacked:
    def __init__(self, tokenizer, to_device=True):
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = "left"  # keep the end (terminal turn, prediction token)
        # Fork change. to_device=False keeps collated tensors on the CPU, so this collate_fn is
        # safe inside forked DataLoader workers (num_workers>0), where CUDA cannot be used. The
        # training loop then does the host-to-device copy (pin_memory + non_blocking).
        # Inference and the default keep to_device=True.
        self.to_device = to_device

    _hdr_len = None

    def _header_len(self):
        """Return the token count of the prompt header kept by LLMKT_KEEP_HEADER=1.

        The header is the KT system prompt (Trace the Ace description) plus the
        ``<|im_start|>user\\n`` marker, rendered with the tokenizer's own chat template. It is
        computed once per process, cached on the class and printed.

        Returns:
            The number of header tokens.
        """
        if LMKTCollatorUnpacked._hdr_len is None:
            from dialogue_kt.prompting import KT_SYSTEM_PROMPT, get_dataset_desc
            class _A: dataset = "traceace"
            sysp = KT_SYSTEM_PROMPT.format(desc=get_dataset_desc(_A()))
            s = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": sysp}, {"role": "user", "content": ""}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            cut = s.index("<|im_start|>user\n") + len("<|im_start|>user\n")
            LMKTCollatorUnpacked._hdr_len = len(self.tokenizer(s[:cut]).input_ids)
            print(f"[keephdr] header tokens = {LMKTCollatorUnpacked._hdr_len}; over-cap prompts keep header + last {MAX_SEQ_LEN - LMKTCollatorUnpacked._hdr_len}")
        return LMKTCollatorUnpacked._hdr_len

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        # Fork change. LLMKT_KEEP_HEADER=1 (set by models/submission/main.py): an over-long
        # prompt keeps its header (system prompt + user marker, H tokens) plus its last
        # MAX_SEQ_LEN - H tokens. A plain left cut, the else branch, would delete the system
        # prompt of the longest dialogues.
        if os.environ.get("LLMKT_KEEP_HEADER") == "1":
            H = self._header_len()
            ids = [self.tokenizer(pr).input_ids for pr in all_prompts]
            ids = [x if len(x) <= MAX_SEQ_LEN else x[:H] + x[-(MAX_SEQ_LEN - H):] for x in ids]
            L = max(len(x) for x in ids); pad = self.tokenizer.pad_token_id
            input_ids = torch.full((len(ids), L), pad, dtype=torch.long)
            attn = torch.zeros((len(ids), L), dtype=torch.long)
            for i, x in enumerate(ids):
                input_ids[i, :len(x)] = torch.tensor(x); attn[i, :len(x)] = 1
            from transformers import BatchEncoding
            prompts_tokenized = BatchEncoding({"input_ids": input_ids, "attention_mask": attn})
        else:
            prompts_tokenized = self.tokenizer(
                all_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_SEQ_LEN,
            )
        num_kcs = torch.LongTensor([len(sample["prompts"]) for sample in batch])
        labels = torch.Tensor([sample["label"] for sample in batch])
        if self.to_device:
            prompts_tokenized = prompts_tokenized.to(device)
            num_kcs = num_kcs.to(device)
            labels = labels.to(device)
        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 1, # Last real token (right-padded); its logits predict True/False (upstream used -2 for its assistant-"\n" prompt)
            "num_kcs": num_kcs,
            "labels": labels,
            "meta_data": batch
        }

_KC_PREFIX_CACHE = {}


def _chat_kc_block(tokenizer, kc: str) -> str:
    """Render one KC as a user turn plus an assistant turn whose newline precedes True/False.

    Packed path only (the shipped model runs with ``--pack_kcs 0``). enable_thinking=False is
    passed for templates that apply it; the fallback is for tokenizers that reject it.

    Args:
        tokenizer: Hugging Face tokenizer with a chat template.
        kc: knowledge-component text.

    Returns:
        The rendered chat text for this KC.
    """
    msgs = [{"role": "user", "content": kc},
            {"role": "assistant", "content": "\n"}]  # newline precedes the True/False prediction
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False)


def _kc_continuation(tokenizer, kc: str) -> str:
    """Render a KC block without its leading BOS and user header, for packed prompts.

    Fork change: upstream split on Llama-3's ``user<|end_header_id|>\\n\\n`` marker. Here a
    sentinel is rendered through the same template and located: everything before it is the
    per-message preamble, identical for every message, so it can be sliced off for any
    template. Packed path only.

    Args:
        tokenizer: Hugging Face tokenizer with a chat template (preamble length cached per
            tokenizer object).
        kc: knowledge-component text.

    Returns:
        A leading space plus the KC's user turn and assistant turn, ready to append.

    Raises:
        RuntimeError: if the template alters the sentinel so it cannot be located.
    """
    key = id(tokenizer)
    if key not in _KC_PREFIX_CACHE:
        sentinel = "\x00KCSENTINEL\x00"
        probe = _chat_kc_block(tokenizer, sentinel)
        if sentinel not in probe:
            raise RuntimeError("chat template mangled the sentinel; cannot locate the user header")
        _KC_PREFIX_CACHE[key] = probe.index(sentinel)
    return " " + _chat_kc_block(tokenizer, kc)[_KC_PREFIX_CACHE[key]:]


def _fit_packed_prompt(tokenizer, sample, dialogue, turn, args, max_len: int):
    """Assemble the packed prompt, dropping the oldest turns until it fits ``max_len`` tokens.

    The packed collator locates the end of the context and each KC's prediction position by
    eos markers, which a token-level left cut would delete. So the prompt is truncated at the
    dialogue level instead: whole turns are dropped from the start and the end is kept (the
    target turn and the KC blocks that carry the prediction positions). A binary search finds
    the smallest number of leading turns to drop. Packed path only.

    Args:
        tokenizer: Hugging Face tokenizer with a chat template.
        sample: the dataset row (used by ``kt_user_prompt``).
        dialogue: annotated turn list of the dialogue.
        turn: the target turn dict (``turn`` number and ``kcs``).
        args: parsed command-line arguments.
        max_len: token budget.

    Returns:
        A tuple ``(prompt, n_dropped)``: the prompt text and the number of leading turns
        dropped (0 if it fitted).
    """
    kc_conts = "".join(_kc_continuation(tokenizer, kc) for kc in turn["kcs"])

    # Fast path: build the full prompt once and return it if it already fits; only over-long
    # prompts go through the binary search below.
    full = tokenizer.apply_chat_template([
        {"role": "system", "content": kt_system_prompt(args)},
        {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], None, args)},
    ], tokenize=False) + kc_conts
    if len(tokenizer(full).input_ids) <= max_len:
        return full, 0

    lo, hi = 0, max(0, len(dialogue) - 1)   # number of leading turns to drop
    keep_from, prompt = 0, None
    while lo <= hi:                          # binary search the smallest safe drop
        mid = (lo + hi) // 2
        trimmed = dialogue[mid:]
        if not any(t["turn"] == turn["turn"] for t in trimmed):
            hi = mid - 1                     # dropped the target turn: too aggressive
            continue
        cand = tokenizer.apply_chat_template([
            {"role": "system", "content": kt_system_prompt(args)},
            {"role": "user", "content": kt_user_prompt(sample, trimmed, turn["turn"], None, args)},
        ], tokenize=False) + kc_conts
        if len(tokenizer(cand).input_ids) <= max_len:
            keep_from, prompt = mid, cand
            hi = mid - 1                     # fits: try dropping fewer
        else:
            lo = mid + 1                     # still too long: drop more
    if prompt is None:                       # cannot fit even minimally; keep the tail
        trimmed = [t for t in dialogue if t["turn"] >= turn["turn"]] or dialogue[-1:]
        keep_from = len(dialogue) - len(trimmed)
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": kt_system_prompt(args)},
            {"role": "user", "content": kt_user_prompt(sample, trimmed, turn["turn"], None, args)},
        ], tokenize=False) + kc_conts
    return prompt, keep_from


class LMKTDatasetPacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        n_trimmed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                # Base prompt + all KC continuations, trimmed at the dialogue level so the
                # collator's eos-based indexing stays intact (see _fit_packed_prompt).
                prompt, dropped = _fit_packed_prompt(
                    tokenizer, sample, dialogue, turn, args, MAX_SEQ_LEN)
                n_trimmed += bool(dropped)
                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")
        if n_trimmed:
            print(f"{n_trimmed} / {len(self.data)} packed prompts trimmed to MAX_SEQ_LEN={MAX_SEQ_LEN}")

class LMKTCollatorPacked:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = prompts_tokenized.input_ids.to(device)
        batch_size, max_seq_len = input_ids.shape
        eos_idxs = [
            (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero().squeeze().cpu()
            for seq_idx in range(batch_size)
        ]
        # Create default lower triangular 3D attention mask
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        # Create default 2D position id matrix
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        # Set attention mask and position ids for each sequence
        for seq_idx in range(batch_size):
            # Get end of context
            context_end_idx = eos_idxs[seq_idx][1]
            # Initialize to no attention to any tokens after context
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            # Update attention mask and position ids for each KC
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                # Set position ids as if KC immediately followed context
                position_ids[seq_idx, start_idx : end_idx + 1] = torch.arange(context_end_idx, context_end_idx + end_idx - start_idx + 1)
                # Set KC attention mask to lower triangular to permit self-attention
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                # Go to next KC
                start_idx = end_idx + 1

        # Get index of token before eos for each KC, pad for easier loss computation
        last_idxs = pad_sequence([idxs[3::2] - 1 for idxs in eos_idxs], batch_first=True)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.unsqueeze(1).to(device), # Add singleton head dimension
            "position_ids": position_ids.to(device),
            "last_idxs": last_idxs,
            "num_kcs": torch.LongTensor([len(sample["kcs"]) for sample in batch]).to(device),
            "labels": torch.Tensor([sample["label"] for sample in batch]).to(device),
            "meta_data": batch
        }

class DKTDataset(DatasetBase):
    def __init__(self, data: pd.DataFrame, kc_dict: Dict[str, int], kc_emb_matrix: torch.Tensor, sbert_model: SentenceTransformer):
        self.data = []
        failed = 0
        num_data_points = 0
        num_correct = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            dialogue_data = {
                "labels": [], "labels_flat": [], "kc_ids": [], "kc_ids_flat": [], "turn_end_idxs": [],
                "teacher_turns": [], "student_turns": [], "kcs": [], "kc_embs": [],
                "dialogue": dialogue, "dialogue_idx": idx
            }
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                dialogue_data["labels"].append(turn["correct"])
                dialogue_data["kc_ids"].append([kc_dict[kc] for kc in turn["kcs"]])
                for kc in turn["kcs"]:
                    dialogue_data["labels_flat"].append(turn["correct"])
                    dialogue_data["kc_ids_flat"].append(kc_dict[kc])
                dialogue_data["turn_end_idxs"].append(len(dialogue_data["kc_ids_flat"]) - 1)
                dialogue_data["teacher_turns"].append(turn["teacher"])
                dialogue_data["student_turns"].append(turn["student"])
                dialogue_data["kcs"].append(turn["kcs"])
                if kc_emb_matrix is not None:
                    dialogue_data["kc_embs"].append(
                        kc_emb_matrix[dialogue_data["kc_ids"][-1]].mean(dim=0)
                    )
            # Add dialogue if at least 2 turns tagged, otherwise nothing to predict
            if len(dialogue_data["labels"]) > 1:
                self.data.append(dialogue_data)
                num_data_points += len(dialogue_data["labels"])
                num_correct += sum(dialogue_data["labels"])
        # Batch encode all dialogue text
        if sbert_model is not None:
            batch_size = 512
            if ALT_ARCH:
                seqs = [dkt_sem_prompt(tt, st, kcs, corr)
                        for dialogue in self.data
                        for tt, st, kcs, corr in zip(dialogue["teacher_turns"], dialogue["student_turns"], dialogue["kcs"], dialogue["labels"])]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["turn_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    turn_counter += seq_len
            else:
                seqs = [turn for dialogue in self.data for turn in dialogue["teacher_turns"]] + [
                        turn for dialogue in self.data for turn in dialogue["student_turns"]]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["teacher_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    stud_start = result_embs.shape[0] // 2
                    dialogue["student_embs"] = result_embs[stud_start + turn_counter : stud_start + turn_counter + seq_len]
                    turn_counter += seq_len
        self.majority_class = 1 if num_correct / num_data_points >= .5 else 0
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Num dialogues: {len(self.data)}, num data points: {num_data_points}, {num_correct} correct")

class DKTCollator:
    def __init__(self, flatten_kcs: bool):
        self.flatten_kcs = flatten_kcs

    def __call__(self, batch):
        labels = pad_sequence(
            [torch.LongTensor(seq["labels"]) for seq in batch],
            batch_first=True, padding_value=-100 # Pad with -100 to ignore loss on padding regions
        )
        # # Fill in KC ids, 2D matrix (length x max num KCs) per sequence
        num_kcs = pad_sequence(
            [torch.LongTensor([len(kc_ids) for kc_ids in seq["kc_ids"]]) for seq in batch],
            batch_first=True, padding_value=1 # Pad with 1 to avoid division by 0
        )
        max_num_kcs = num_kcs.max()
        kc_ids = torch.zeros((*num_kcs.shape, max_num_kcs), dtype=torch.long)
        for seq_idx, seq in enumerate(batch):
            for turn_idx, turn_kc_ids in enumerate(seq["kc_ids"]):
                kc_ids[seq_idx, turn_idx, :len(turn_kc_ids)] = torch.LongTensor(turn_kc_ids)

        result = {
            "labels": labels.to(device),
            "kc_ids": kc_ids.to(device),
            "num_kcs": num_kcs.to(device)
        }

        if self.flatten_kcs:
            # Add flattened versions of KC ids and labels for unrolled model input
            kc_ids_flat = pad_sequence([torch.LongTensor(seq["kc_ids_flat"]) for seq in batch], batch_first=True)
            labels_flat = pad_sequence([torch.LongTensor(seq["labels_flat"]) for seq in batch], batch_first=True)
            turn_end_idxs = pad_sequence([torch.LongTensor(seq["turn_end_idxs"]) for seq in batch], batch_first=True)
            result = {
                **result,
                "labels_flat": labels_flat.to(device),
                "kc_ids_flat": kc_ids_flat.to(device),
                "turn_end_idxs": turn_end_idxs.to(device)
            }

        # Add text embeddings for DKT-Sem
        if batch[0]["kc_embs"] and not ALT_ARCH:
            kc_embs = pad_sequence([torch.stack(seq["kc_embs"]) for seq in batch], batch_first=True)
            teacher_embs = pad_sequence([seq["teacher_embs"] for seq in batch], batch_first=True)
            student_embs = pad_sequence([seq["student_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "kc_embs": kc_embs,
                "teacher_embs": teacher_embs,
                "student_embs": student_embs
            }
        elif "turn_embs" in batch[0]:
            turn_embs = pad_sequence([seq["turn_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "turn_embs": turn_embs
            }

        return result

def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool, sampler=None,
                   num_workers: int = 0, pin_memory: bool = False):
    # Fork change. sampler is used by DDP training (DistributedSampler) and overrides shuffle.
    # num_workers>0 runs collation in worker processes (requires a CPU-tensor collator, i.e.
    # to_device=False, with the loop doing the host-to-device copy).
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size,
                      shuffle=shuffle if sampler is None else False, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0))


def get_length_grouped_indices(lengths, batch_size, generator=None, mega_batch_mult=None):
    """Order sample indices so that batches have similar lengths (HF `group_by_length` recipe).

    Indices are shuffled globally, then sorted by length within mega-batches, so batches come
    out length-homogeneous while the epoch stays globally shuffled and is re-drawn every epoch.
    Used only by DistributedLengthGroupedSampler.

    Args:
        lengths: per-sample length proxy (e.g. ``LMKTDatasetUnpacked.lengths``).
        batch_size: per-rank batch size.
        generator: optional ``torch.Generator`` for the shuffle.
        mega_batch_mult: batches per mega-batch (default: min(len // (4 * batch_size), 50),
            at least 1).

    Returns:
        A flat list of sample indices.
    """
    import math  # noqa: F401  (kept local so the module import stays cheap)
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        mega_batch_mult = max(mega_batch_mult, 1)
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i:i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
    # sort each mega-batch by length (desc); mega-batch *order* stays as-shuffled -> no curriculum
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]
    return [i for mb in megabatches for i in mb]


class DistributedLengthGroupedSampler(torch.utils.data.Sampler):
    """DDP sampler whose per-step batches have similar lengths on every rank.

    Each rank takes a strided subsample of a locally length-sorted order (from
    ``get_length_grouped_indices``), so step k has similar lengths on all ranks and the N-way
    gradient allreduce does not wait for one slow rank on variable-length dialogues. The order
    is padded by wrap-around so every rank gets exactly the same number of samples; a rank that
    ran short would leave the allreduce hanging. Enabled with LLMKT_GROUP_BY_LENGTH=1; by
    default DDP training uses DistributedSampler.

    Args:
        lengths: per-sample length proxy.
        batch_size: per-rank batch size.
        num_replicas: world size.
        rank: this process's rank.
        seed: base seed; the epoch number is added (see ``set_epoch``).
    """
    def __init__(self, lengths, batch_size, num_replicas, rank, seed=221):
        import math
        self.lengths = lengths
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(lengths) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)
        # pad (wrap-around) so len == total_size and every rank gets exactly num_samples
        padding = self.total_size - len(indices)
        indices = (indices + indices[:padding]) if padding > 0 else indices[:self.total_size]
        assert len(indices) == self.total_size
        # strided subsample: rank r -> positions r, r+W, r+2W, ...  Because the global order is
        # locally length-sorted, each rank's consecutive batch draws from the same length window,
        # and rank r's step-k batch aligns in length with every other rank's step-k batch.
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
