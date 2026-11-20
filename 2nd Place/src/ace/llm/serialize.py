"""Canonical (transcript, objective) -> LLM prompt serialization (E13).

One shared implementation for LoRA training, vLLM CV scoring and the runtime
main.py, so every model sees byte-identical inputs.

Design (EXPERIMENTS.md E13):
  - transcript FIRST, objective + Yes/No question LAST — recency for the model,
    and same-session rows share the transcript prefix (vLLM prefix caching).
  - token budget `max_input_tokens`: default 8192 covers 99.6% of transcripts
    fully (EDA.md §2: median 4.9K, p99 7.8K, max 11.5K tokens); 16384 covers 100%
    at ~1-2% extra total tokens (only the p99+ tail lengthens) but requires
    max_model_len >= 16384+GEN_MARGIN on the inference side.
  - overflow -> MIDDLE truncation: keep the opening (lesson context) and the tail
    (wrap-up / assessment evidence), drop utterances from the middle behind an
    explicit omission marker.
  - Qwen3 chat template: thinking disabled (enable_thinking=False) — we read the
    Yes/No answer token directly.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

SYSTEM_PROMPT = ("You are an expert math-tutoring evaluator. You read a transcript of "
                 "a one-on-one online lesson and judge the student's mastery.")
QUESTION = ("Based on this lesson, will the student answer a post-lesson assessment "
            "question on this learning objective correctly? Answer with Yes or No only.")
OMIT_MARKER = "[... {n} utterances omitted ...]"
HEAD_FRAC = 0.30      # share of the transcript budget kept from the start
GEN_MARGIN = 8        # tokens reserved for generation prompt + answer
YES, NO = "Yes", "No"


SPEAKER_TAG_RE = re.compile(r"\[speaker:\s*([^\]]+)\]", re.IGNORECASE)
BG_MIN_WORDS = 15  # v2: background segments shorter than this are treated as noise


def _speaker_of(tag: str) -> str:
    t = tag.strip().lower()
    if t == "tutor":
        return "tutor"
    if t == "background":
        return "background"
    return "student"  # one-on-one lessons: any named speaker is the student


def read_dialogue(path: str | Path, version: int = 1) -> list[str]:
    """Transcript CSV -> ['Tutor: ...', 'Student: ...'] lines.

    version=1 (default, matches all models trained before 2026-07-05): the CSV
    `role` column is trusted verbatim; background rows dropped; embedded
    [Speaker:...] tags left inside the content.

    version=2 (diarization repair): the role column is unreliable at the row
    level (~1.1% of rows carry embedded [Speaker:...] tags contradicting it,
    39% of sessions affected; see EXPERIMENTS.md role-reliability scan). Rows
    are split on embedded tags and each segment is reattributed to the tagged
    speaker (tags stripped from the text). Background segments are kept only
    when substantial (>= BG_MIN_WORDS words, often real dialogue), prefixed
    'Background:'. A model must be scored/deployed with the SAME version it
    was trained with (recorded in the fold's config.json).
    """
    lines: list[str] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            role = (row.get("role") or "").strip().lower()
            content = (row.get("content") or "").strip()
            if not content:
                continue
            if version == 1:
                if role in ("tutor", "student"):
                    lines.append(f"{'Tutor' if role == 'tutor' else 'Student'}: {content}")
                continue
            if role not in ("tutor", "student", "background"):
                continue
            segs, pos, speaker = [], 0, role
            for m in SPEAKER_TAG_RE.finditer(content):
                seg = content[pos:m.start()].strip()
                if seg:
                    segs.append((speaker, seg))
                speaker, pos = _speaker_of(m.group(1)), m.end()
            tail = content[pos:].strip()
            if tail:
                segs.append((speaker, tail))
            for sp, text in segs:
                if sp == "background":
                    if len(text.split()) >= BG_MIN_WORDS:
                        lines.append(f"Background: {text}")
                else:
                    lines.append(f"{'Tutor' if sp == 'tutor' else 'Student'}: {text}")
    return lines


def _template_overhead(tokenizer, objective: str) -> int:
    """Token count of the full prompt with an empty transcript."""
    return len(apply_template(tokenizer, _user_message("", objective)))


def _user_message(dialogue: str, objective: str) -> str:
    return (f"Transcript of the lesson:\n{dialogue}\n\n"
            f"Learning objective assessed: {objective}\n\n{QUESTION}")


def truncate_middle(lines: list[str], line_tokens: list[int], budget: int,
                    head_frac: float = HEAD_FRAC) -> list[str]:
    """Keep head (``head_frac`` of budget) + tail, drop the middle behind a marker.

    ``head_frac`` defaults to the module-level HEAD_FRAC, so every existing caller
    (and every model trained before this parameter existed) is bit-for-bit unchanged.
    It is exposed only so the ModernBERT reader can sweep it without forking this
    function.
    """
    total = sum(line_tokens)
    if total <= budget:
        return lines
    head_budget = int(budget * head_frac)
    head_end, acc = 0, 0
    for i, t in enumerate(line_tokens):
        if acc + t > head_budget:
            break
        acc += t
        head_end = i + 1
    tail_budget = budget - acc - 16  # marker allowance
    tail_start, acc_t = len(lines), 0
    for i in range(len(lines) - 1, head_end - 1, -1):
        if acc_t + line_tokens[i] > tail_budget:
            break
        acc_t += line_tokens[i]
        tail_start = i
    omitted = tail_start - head_end
    if omitted <= 0:
        return lines
    return lines[:head_end] + [OMIT_MARKER.format(n=omitted)] + lines[tail_start:]


def build_messages(tokenizer, lines: list[str], objective: str,
                   max_input_tokens: int = 8192) -> list[dict]:
    """Chat messages for one (transcript, objective) row, within the token budget."""
    overhead = _template_overhead(tokenizer, objective)
    budget = max_input_tokens - overhead - GEN_MARGIN
    line_tokens = [len(tokenizer(ln, add_special_tokens=False).input_ids) + 1
                   for ln in lines]
    kept = truncate_middle(lines, line_tokens, budget)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message("\n".join(kept), objective)}]


def _call_chat_template(tokenizer, msgs, base_kw):
    """apply_chat_template, degrading only over OPTIONAL kwargs some tokenizers lack.

    Two optional kwargs, most-preferred first:
      * ``enable_thinking=False`` — Qwen3/Qwen3.5 non-thinking mode. Templates that
        predate the flag reject it; that has always been tolerated here.
      * ``return_dict=False`` — Transformers <=4.x already defaults to this when
        ``tokenize=True``, so passing it explicitly is a no-op there and changes
        nothing about the 4.57.6 competition runtime. Transformers 5 flipped the
        default to a BatchEncoding, and this asks for the historical form back.

    Tokenizers vary independently of package version (local templates, third-party
    implementations, test fakes), so support is discovered by trying rather than by
    branching on ``transformers.__version__``. Only TypeError/ValueError from an
    unsupported kwarg is absorbed; if every combination fails the last error is
    raised so a genuine template problem still surfaces.
    """
    attempts = ({"enable_thinking": False, "return_dict": False},
                {"enable_thinking": False},
                {"return_dict": False},
                {})
    last: Exception | None = None
    for extra in attempts:
        try:
            return tokenizer.apply_chat_template(msgs, **base_kw, **extra)
        except (TypeError, ValueError) as e:
            last = e
    raise last  # type: ignore[misc]


def _normalize_template_token_ids(result) -> list[int]:
    """Coerce any apply_chat_template return shape to a flat list[int].

    ACE always templates exactly ONE prompt, so anything carrying several sequences
    is an error rather than something to flatten — silently concatenating two prompts
    would produce a plausible-looking but meaningless input.

    Handled: a flat list/tuple of ids; a dict/BatchEncoding (take ``input_ids``); a
    tensor/array exposing ``.tolist()``; and a single leading batch dimension.
    """
    obj = result
    if hasattr(obj, "keys") and not isinstance(obj, (list, tuple)):  # dict / BatchEncoding
        try:
            obj = obj["input_ids"]
        except (KeyError, TypeError):
            raise TypeError(
                f"apply_template expected token IDs but the tokenizer returned "
                f"{type(result).__name__} without an 'input_ids' entry; expected "
                f"list[int] or an object containing input_ids") from None
    if hasattr(obj, "tolist") and not isinstance(obj, (list, tuple)):  # tensor / ndarray
        obj = obj.tolist()
    if not isinstance(obj, (list, tuple)):
        raise TypeError(
            f"apply_template expected token IDs but the tokenizer returned "
            f"{type(result).__name__}; expected list[int] or an object containing "
            f"input_ids")
    ids = list(obj)
    if ids and all(isinstance(x, (list, tuple)) for x in ids):  # one batch dimension
        if len(ids) != 1:
            raise ValueError(
                f"apply_template templates a single prompt but the tokenizer returned "
                f"{len(ids)} sequences; refusing to flatten them together")
        ids = list(ids[0])
    if not ids:
        raise ValueError("apply_template produced an empty prompt")
    out: list[int] = []
    for x in ids:
        # bool is a subclass of int and implements __index__, so exclude it explicitly.
        # __index__ accepts numpy/torch integer scalars without importing either.
        if isinstance(x, bool) or not hasattr(x, "__index__"):
            raise TypeError(
                f"apply_template expected integer token IDs but found "
                f"{type(x).__name__} ({x!r}); expected list[int] or an object "
                f"containing input_ids")
        out.append(int(x.__index__()))
    return out


def apply_template(tokenizer, messages_or_user, add_generation_prompt: bool = True) -> list[int]:
    """Chat template -> flat list[int] of token ids; disables thinking mode when supported.

    CONTRACT: always a flat Python ``list[int]``, under Transformers 4.57.6 and 5.x
    alike. Transformers <=4.x commonly returns list[int]; Transformers 5 may return a
    BatchEncoding by default. ACE normalizes both because training, vLLM scoring and
    the runtime ``score_worker`` all depend on that one stable representation —
    ``score_worker`` hands it straight to vLLM as ``prompt_token_ids``, and
    ``objective_context``/``_template_overhead`` use ``len()`` of it as an actual token
    count for budgeting. A BatchEncoding silently made that length 2 (the key count).

    Only the API's return REPRESENTATION is normalized. The rendered prompt text,
    token ids, ordering, special tokens, generation prompt and thinking-mode semantics
    are untouched: nothing here decodes, re-tokenizes, pads or truncates.
    """
    msgs = (messages_or_user if isinstance(messages_or_user, list)
            else [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": messages_or_user}])
    kw = dict(add_generation_prompt=add_generation_prompt, tokenize=True)
    return _normalize_template_token_ids(_call_chat_template(tokenizer, msgs, kw))


def answer_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Candidate single-token ids for Yes / No (with and without leading space)."""
    def ids(word):
        out = []
        for v in (word, " " + word, word.lower(), " " + word.lower()):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                out.append(enc[0])
        return sorted(set(out))
    y, n = ids(YES), ids(NO)
    assert y and n, "tokenizer has no single-token Yes/No — adjust answer words"
    return y, n
