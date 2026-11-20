"""Objective-aware head/chunks/tail prompt construction for transcript readers.

The standard ACE E13 serializer keeps the whole transcript whenever it fits. This
module intentionally implements a *compressed key-moment reader* for a complementary
model:

* sessions that fit the configured evidence budget are kept whole;
* longer sessions keep a compact lesson opening and ending;
* BGE-localised semantic windows are expanded through neighbouring utterances until
  their allocated model-token budgets are used;
* every selected excerpt remains in chronological order and utterances are never
  duplicated.

A semantic context index is produced offline by ``prepare_objective_contexts.py``.
The index uses only the current response's objective and transcript, never labels or
other test rows, so the same code can be used at inference without cross-sample
features.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

from ace.llm.serialize import GEN_MARGIN, QUESTION, apply_template, build_messages

OBJECTIVE_SYSTEM_PROMPT = (
    "You are an expert math-tutoring evaluator. You judge a student's mastery from "
    "selected, chronologically ordered excerpts of a one-on-one lesson. Focus on the "
    "student's reasoning, corrections, independence, and final understanding."
)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_VERSION = 2
SECTION_SAFETY_TOKENS = 96
_WORD_RE = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", re.IGNORECASE)
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "up", "using", "with", "within", "each", "into", "than", "numbers", "number",
}


@dataclass(frozen=True)
class ContextWindow:
    start: int
    end: int
    score: float = 0.0

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid context window [{self.start}, {self.end})")


@dataclass(frozen=True)
class ContextPlan:
    response_id: str
    session_id: str
    serialize: int
    windows: tuple[ContextWindow, ...]


def context_meta_path(index_path: str | Path) -> Path:
    p = Path(index_path)
    return p.with_suffix(p.suffix + ".meta.json")


def repair_trailing_partial_jsonl(path: str | Path) -> bool:
    """Remove only an invalid, unterminated final JSONL fragment.

    The context builder flushes complete lines after every session batch. If a job is
    killed in the middle of the final write, this makes the next run resumable without
    discarding any complete response rows. A valid final line without a newline is kept.
    """
    p = Path(path).expanduser()
    if not p.exists() or p.stat().st_size == 0:
        return False
    raw = p.read_bytes()
    if raw.endswith(b"\n"):
        return False
    last_nl = raw.rfind(b"\n")
    tail = raw[last_nl + 1:]
    try:
        json.loads(tail.decode("utf-8"))
        return False
    except (UnicodeDecodeError, json.JSONDecodeError):
        keep = raw[:last_nl + 1] if last_nl >= 0 else b""
        p.write_bytes(keep)
        return True


def load_context_index(path: str | Path) -> dict[str, ContextPlan]:
    """Load JSONL context plans keyed by response_id, rejecting duplicates."""
    out: dict[str, ContextPlan] = {}
    with Path(path).expanduser().open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            rid = str(row["response_id"])
            if rid in out:
                raise ValueError(f"duplicate response_id {rid!r} at line {lineno}")
            wins = tuple(ContextWindow(int(w[0]), int(w[1]), float(w[2]))
                         for w in row.get("windows", []))
            out[rid] = ContextPlan(
                response_id=rid,
                session_id=str(row["session_id"]),
                serialize=int(row.get("serialize", 1)),
                windows=wins,
            )
    return out


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def candidate_windows(lines: Sequence[str], window_words: int = 240,
                      stride_words: int = 120, min_words: int = 40) -> list[tuple[int, int]]:
    """Create deterministic contiguous utterance windows using word budgets."""
    if not lines:
        return []
    if window_words <= 0 or stride_words <= 0:
        raise ValueError("window_words and stride_words must be positive")
    wc = [max(1, count_words(x)) for x in lines]
    n = len(lines)
    starts = [0]
    acc = 0
    for i, w in enumerate(wc):
        acc += w
        if acc >= stride_words:
            nxt = i + 1
            if nxt < n:
                starts.append(nxt)
            acc = 0
    windows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start in starts:
        total = 0
        end = start
        while end < n and total < window_words:
            total += wc[end]
            end += 1
        if end <= start:
            continue
        if total < min_words and windows:
            # Attach a tiny final fragment to the preceding candidate instead of
            # creating a low-information standalone passage.
            continue
        pair = (start, end)
        if pair not in seen:
            windows.append(pair)
            seen.add(pair)
    if not windows:
        windows = [(0, n)]
    elif windows[-1][1] < n:
        # Guarantee coverage of the lesson ending as a retrieval candidate too.
        start = max(0, n - max(1, windows[-1][1] - windows[-1][0]))
        pair = (start, n)
        if pair not in seen:
            windows.append(pair)
    return windows


def overlap_fraction(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    return inter / max(1, min(a[1] - a[0], b[1] - b[0]))


def select_ranked_windows(windows: Sequence[tuple[int, int]], scores: Sequence[float],
                          top_k: int = 8, max_overlap: float = 0.55) -> list[ContextWindow]:
    """Keep high-scoring, not-near-duplicate windows with deterministic tie breaks."""
    if len(windows) != len(scores):
        raise ValueError("windows/scores length mismatch")
    order = sorted(range(len(windows)), key=lambda i: (-float(scores[i]), windows[i][0], windows[i][1]))
    chosen: list[ContextWindow] = []
    for i in order:
        pair = windows[i]
        if any(overlap_fraction(pair, (c.start, c.end)) > max_overlap for c in chosen):
            continue
        chosen.append(ContextWindow(pair[0], pair[1], float(scores[i])))
        if len(chosen) >= top_k:
            break
    if not chosen and windows:
        i = order[0]
        chosen = [ContextWindow(windows[i][0], windows[i][1], float(scores[i]))]
    return chosen


def _lex_tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if len(t) > 1 and t not in _STOP}


def lexical_fallback_plan(lines: Sequence[str], objective: str, response_id: str = "",
                          session_id: str = "", serialize: int = 1) -> ContextPlan:
    """Label-free emergency plan used only when no semantic index row is available."""
    windows = candidate_windows(lines)
    q = _lex_tokens(objective)
    scores = []
    for a, b in windows:
        p = _lex_tokens(" ".join(lines[a:b]))
        overlap = len(q & p) / max(1.0, math.sqrt(max(1, len(q)) * max(1, len(p))))
        # A tiny numerical/mathematical content bonus breaks all-zero ties without
        # pretending to be semantic retrieval.
        text = " ".join(lines[a:b])
        bonus = min(0.03, 0.002 * len(re.findall(r"\d|[=+×÷/%]", text)))
        scores.append(overlap + bonus)
    return ContextPlan(str(response_id), str(session_id), int(serialize),
                       tuple(select_ranked_windows(windows, scores)))


def _token_lengths(tokenizer, lines: Sequence[str]) -> list[int]:
    return [len(tokenizer.encode(x, add_special_tokens=False)) + 1 for x in lines]


def _prefix_indices(lengths: Sequence[int], budget: int, blocked: set[int] | None = None) -> list[int]:
    blocked = blocked or set()
    out: list[int] = []
    used = 0
    for i, n in enumerate(lengths):
        if i in blocked:
            continue
        if out and used + n > budget:
            break
        if not out and n > budget:
            out.append(i)
            break
        out.append(i)
        used += n
    return out


def _suffix_indices(lengths: Sequence[int], budget: int, blocked: set[int] | None = None) -> list[int]:
    blocked = blocked or set()
    out: list[int] = []
    used = 0
    for i in range(len(lengths) - 1, -1, -1):
        if i in blocked:
            continue
        n = lengths[i]
        if out and used + n > budget:
            break
        if not out and n > budget:
            out.append(i)
            break
        out.append(i)
        used += n
    return sorted(out)


def _centered_indices(lengths: Sequence[int], start: int, end: int, budget: int,
                      blocked: set[int]) -> list[int]:
    """Expand a semantic anchor through unblocked neighbours up to ``budget``.

    ``start:end`` is only the BGE localisation window. The returned excerpt is allowed
    to grow beyond that window, which is essential because retrieval windows are kept
    deliberately narrow for localisation while the model-facing evidence chunks are
    much larger. Expansion stops at already selected head/tail/chunk utterances.
    """
    n_lines = len(lengths)
    start, end = max(0, start), min(n_lines, end)
    if start >= end or budget <= 0:
        return []

    # Maximal unblocked runs across the *whole* lesson. Keep only runs touching the
    # semantic window so a high-scoring retrieval hit remains the anchor.
    runs: list[list[int]] = []
    current: list[int] = []
    for i in range(n_lines):
        if i in blocked:
            if current:
                runs.append(current)
                current = []
        else:
            current.append(i)
    if current:
        runs.append(current)
    touching = [r for r in runs if r[0] < end and r[-1] + 1 > start]
    if not touching:
        return []

    target = (start + end - 1) / 2.0
    def run_key(r: list[int]):
        overlap = max(0, min(r[-1] + 1, end) - max(r[0], start))
        center = (r[0] + r[-1]) / 2.0
        return (-overlap, abs(center - target), -len(r), r[0])

    run = min(touching, key=run_key)
    anchor_candidates = [j for j, i in enumerate(run) if start <= i < end]
    if not anchor_candidates:
        return []
    center_pos = min(anchor_candidates, key=lambda j: (abs(run[j] - target), run[j]))

    chosen = [run[center_pos]]
    used = lengths[run[center_pos]]
    left, right = center_pos - 1, center_pos + 1
    while left >= 0 or right < len(run):
        candidates = []
        if left >= 0:
            candidates.append((abs(run[left] - target), 0, left))
        if right < len(run):
            candidates.append((abs(run[right] - target), 1, right))
        _, side, pos = min(candidates)
        i = run[pos]
        if used + lengths[i] > budget:
            # This utterance may be unusually long; try the other side before stopping.
            if side == 0:
                left = -1
            else:
                right = len(run)
            continue
        chosen.append(i)
        used += lengths[i]
        if side == 0:
            left -= 1
        else:
            right += 1
    return sorted(chosen)


def _objective_user_message(objective: str, sections: Sequence[tuple[str, Sequence[str]]]) -> str:
    # Evidence comes first so rows from the same session can share the opening prefix in
    # vLLM. The objective stays immediately before the decision question, where it is
    # also less likely to be diluted by a long transcript.
    pieces = ["Selected evidence from the lesson (chronological excerpts):"]
    for title, lines in sections:
        if lines:
            pieces.extend([f"[{title}]", "\n".join(lines)])
    pieces.extend([f"Learning objective assessed: {objective}", QUESTION])
    return "\n\n".join(pieces)


def _messages(objective: str, sections: Sequence[tuple[str, Sequence[str]]]) -> list[dict]:
    return [{"role": "system", "content": OBJECTIVE_SYSTEM_PROMPT},
            {"role": "user", "content": _objective_user_message(objective, sections)}]


def select_objective_sections(tokenizer, lines: Sequence[str], objective: str,
                              plan: ContextPlan | None, max_input_tokens: int = 8192,
                              full_context_threshold: int = 0,
                              n_chunks: int = 2, head_frac: float = 0.16,
                              tail_frac: float = 0.34) -> list[tuple[str, list[str]]]:
    """Select exact excerpts under the model-token budget.

    The semantic plan ranks candidate spans. Token budgeting remains tokenizer-specific
    and is therefore performed here, identically for training and scoring.
    """
    if not lines:
        return [("COMPLETE LESSON", ["(No usable tutor/student utterances.)"])]
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if head_frac < 0 or tail_frac < 0 or head_frac + tail_frac >= 0.90:
        raise ValueError("invalid head/tail fractions")

    lengths = _token_lengths(tokenizer, lines)
    total = sum(lengths)
    empty_overhead = len(apply_template(tokenizer, _messages(objective, [])))
    evidence_budget = max_input_tokens - empty_overhead - GEN_MARGIN - SECTION_SAFETY_TOKENS
    if evidence_budget < 256:
        raise ValueError(f"max_input_tokens={max_input_tokens} leaves too little evidence budget")

    complete_limit = evidence_budget if full_context_threshold <= 0 else min(
        evidence_budget, full_context_threshold)
    if total <= complete_limit:
        return [("COMPLETE LESSON", list(lines))]

    head_budget = max(96, int(evidence_budget * head_frac))
    tail_budget = max(128, int(evidence_budget * tail_frac))
    chunk_budget = max(128, (evidence_budget - head_budget - tail_budget) // n_chunks)

    head_idx = _prefix_indices(lengths, head_budget)
    blocked = set(head_idx)
    tail_idx = _suffix_indices(lengths, tail_budget, blocked)
    blocked.update(tail_idx)

    if plan is None or not plan.windows:
        plan = lexical_fallback_plan(lines, objective)
    ranked = list(plan.windows)
    selected: list[tuple[int, int, list[int]]] = []
    for w in ranked:
        original_pair = (w.start, w.end)
        if any(overlap_fraction(original_pair, (a, b)) > 0.45 for a, b, _ in selected):
            continue
        idx = _centered_indices(lengths, w.start, w.end, chunk_budget, blocked)
        if not idx:
            continue
        # Reject a candidate that contributes almost no new transcript extent.
        pair = (idx[0], idx[-1] + 1)
        if any(overlap_fraction(pair, (a, b)) > 0.60 for a, b, _ in selected):
            continue
        selected.append((pair[0], pair[1], idx))
        blocked.update(idx)
        if len(selected) >= n_chunks:
            break

    # Deterministic fallback when semantic candidates were swallowed by head/tail.
    if len(selected) < n_chunks:
        anchors = [int((j + 1) * len(lines) / (n_chunks + 1)) for j in range(n_chunks)]
        radius = max(1, len(lines) // (2 * (n_chunks + 1)))
        for anchor in anchors:
            idx = _centered_indices(lengths, anchor - radius, anchor + radius + 1,
                                    chunk_budget, blocked)
            if not idx:
                continue
            selected.append((idx[0], idx[-1] + 1, idx))
            blocked.update(idx)
            if len(selected) >= n_chunks:
                break

    selected.sort(key=lambda x: (x[0], x[1]))
    sections: list[tuple[str, list[str]]] = []
    if head_idx:
        sections.append(("LESSON OPENING", [lines[i] for i in head_idx]))
    for j, (_, _, idx) in enumerate(selected, 1):
        sections.append((f"OBJECTIVE-RELEVANT MOMENT {j}", [lines[i] for i in idx]))
    if tail_idx:
        sections.append(("LESSON ENDING", [lines[i] for i in tail_idx]))

    # Exact final guard. Marker/template tokenization is not perfectly additive, so
    # shave the largest section until the generated prompt is within budget.
    mutable = [[title, list(xs)] for title, xs in sections]
    while len(apply_template(tokenizer, _messages(objective, mutable))) > max_input_tokens:
        candidates = [(sum(_token_lengths(tokenizer, xs)), i)
                      for i, (_, xs) in enumerate(mutable) if len(xs) > 1]
        if not candidates:
            break
        _, i = max(candidates)
        title, xs = mutable[i]
        if title == "LESSON ENDING":
            xs.pop(0)
        else:
            xs.pop()
    final = [(str(title), list(xs)) for title, xs in mutable if xs]
    n_final = len(apply_template(tokenizer, _messages(objective, final)))
    if n_final > max_input_tokens:
        raise RuntimeError(f"objective-aware prompt still exceeds budget: {n_final}>{max_input_tokens}")
    return final


def build_objective_messages(tokenizer, lines: Sequence[str], objective: str,
                             plan: ContextPlan | None, max_input_tokens: int = 8192,
                             full_context_threshold: int = 0,
                             n_chunks: int = 2, head_frac: float = 0.16,
                             tail_frac: float = 0.34) -> list[dict]:
    sections = select_objective_sections(
        tokenizer, lines, objective, plan, max_input_tokens=max_input_tokens,
        full_context_threshold=full_context_threshold, n_chunks=n_chunks,
        head_frac=head_frac, tail_frac=tail_frac,
    )
    return _messages(objective, sections)


def build_prompt_messages(tokenizer, lines: Sequence[str], objective: str,
                          max_input_tokens: int = 8192, prompt_mode: str = "standard",
                          response_id: str = "", session_id: str = "",
                          context_index: Mapping[str, ContextPlan] | None = None,
                          require_context_index: bool = False,
                          context_full_threshold: int = 0,
                          context_chunks: int = 2,
                          context_head_frac: float = 0.16,
                          context_tail_frac: float = 0.34) -> list[dict]:
    """Shared train/score dispatch; default behavior is byte-compatible with E13."""
    if prompt_mode == "standard":
        return build_messages(tokenizer, list(lines), objective, max_input_tokens)
    if prompt_mode != "objective_chunks":
        raise ValueError(f"unknown prompt_mode={prompt_mode!r}")
    plan = context_index.get(str(response_id)) if context_index is not None else None
    if plan is not None:
        if plan.session_id and str(session_id) and plan.session_id != str(session_id):
            raise ValueError(f"context-index session mismatch for response_id={response_id}")
    elif require_context_index:
        raise KeyError(f"response_id={response_id} is absent from the required context index")
    return build_objective_messages(
        tokenizer, lines, objective, plan, max_input_tokens=max_input_tokens,
        full_context_threshold=context_full_threshold, n_chunks=context_chunks,
        head_frac=context_head_frac, tail_frac=context_tail_frac,
    )
