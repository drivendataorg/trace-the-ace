"""Structured transcript features for the ACE ``tabular_v2`` reader.

The module intentionally produces scalar, interpretable features that can be consumed by
LightGBM.  It does not use labels and is therefore safe to run identically on train and test.

Feature families
----------------
* session structure / transcript quality
* productive mathematical talk
* tutor -> student transitions
* first/middle/final-third progression
* student/tutor-specific objective lexical coverage
* optional student-only BGE objective coverage

The heuristics are deliberately simple and deterministic.  They are not intended to be
perfect dialogue-act classifiers; the downstream GBM can learn which noisy indicators are
useful.
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Keep these definitions independent from ace.transcripts so this module can be copied into
# a submission package without importing heavy training-only code.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)?|[%+\-*/=<>]", re.I)
ALPHA_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?%?(?!\w)")
SPEAKER_TAG_RE = re.compile(r"\[\s*speaker\s*:\s*(tutor|student|background)\s*\]", re.I)
BRACKET_RE = re.compile(r"\[[^\]]{1,80}\]")

STOPWORDS = set(
    """a an and are as at be been being by can could did do does for from had has have he
    her him his how i if in into is it its me my not of on or our she so that the their them
    then there they this to up us was we were what when which who will with would you your
    yours it's don't i'm that's""".split()
)

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
    "half", "halves", "quarter", "quarters", "third", "thirds", "fourth", "fourths",
    "fifth", "fifths", "sixth", "sixths", "seventh", "sevenths", "eighth", "eighths",
    "ninth", "ninths", "tenth", "tenths", "percent", "percentage",
}

MATH_WORDS = NUMBER_WORDS | {
    "add", "added", "adding", "addition", "subtract", "subtracted", "subtracting",
    "subtraction", "minus", "plus", "multiply", "multiplied", "multiplying",
    "multiplication", "times", "divide", "divided", "division", "fraction", "fractions",
    "decimal", "decimals", "numerator", "denominator", "equation", "equals", "equal",
    "greater", "less", "ratio", "proportion", "multiple", "factor", "factors", "prime",
    "square", "squared", "cube", "cubed", "angle", "degrees", "perimeter", "area",
    "volume", "length", "width", "height", "shape", "triangle", "rectangle", "circle",
    "integer", "integers", "positive", "negative", "sequence", "pattern", "place",
    "value", "digit", "digits", "round", "rounding", "estimate", "difference", "sum",
    "product", "quotient", "remainder", "mean", "median", "mode", "probability",
}

REASON_PATTERNS = (
    "because", "so ", "so,", "therefore", "then", "if ", "which means", "that means",
    "would make", "would be", "i know", "i think", "first", "next", "after that",
    "the reason", "as a result", "since ", "to get", "we need", "i need",
)
SELF_CORRECT_PATTERNS = (
    "wait", "actually", "i mean", "sorry", "no,", "no ", "oh", "my mistake",
    "i got", "let me", "scratch that", "not that",
)
CONFUSION_PATTERNS = (
    "i don't know", "i dont know", "not sure", "i'm not sure", "im not sure", "confused",
    "i don't understand", "i dont understand", "huh", "what?", "maybe", "i forgot",
)
TUTOR_CORRECTION_PATTERNS = (
    "not quite", "not correct", "isn't right", "is not right", "wrong", "try again",
    "check that", "check your", "mistake", "should be", "remember that", "be careful",
    "have another look", "look again", "not the answer",
)
TUTOR_PRAISE_PATTERNS = (
    "well done", "good job", "great job", "excellent", "brilliant", "perfect", "correct",
    "that's right", "that is right", "very good", "nice work", "good thinking", "exactly",
)
TUTOR_HINT_PATTERNS = (
    "think about", "what happens", "what could", "how could", "can you", "try to",
    "have a look", "look at", "remember", "use the", "start with", "what do you notice",
    "what would", "which operation", "what is the first",
)
TUTOR_CHECK_PATTERNS = (
    "does that make sense", "do you understand", "are you happy", "can you explain",
    "can you tell me why", "why is that", "how do you know", "are you sure",
)
BACKCHANNELS = {
    "yes", "yeah", "yep", "no", "okay", "ok", "right", "mhm", "mm", "uh huh",
    "uh-huh", "hmm", "sure", "alright", "fine",
}

def _compile_phrases(patterns: Sequence[str]) -> re.Pattern[str]:
    # Longest first avoids a short alternative consuming a longer phrase.  These are
    # indicator searches, so exact match span is irrelevant.
    parts = sorted((re.escape(p.strip()) for p in patterns if p.strip()), key=len, reverse=True)
    return re.compile("|".join(parts), re.I)


REASON_RE = _compile_phrases(REASON_PATTERNS)
SELF_CORRECT_RE = _compile_phrases(SELF_CORRECT_PATTERNS)
CONFUSION_RE = _compile_phrases(CONFUSION_PATTERNS)
TUTOR_CORRECTION_RE = _compile_phrases(TUTOR_CORRECTION_PATTERNS)
TUTOR_PRAISE_RE = _compile_phrases(TUTOR_PRAISE_PATTERNS)
TUTOR_HINT_RE = _compile_phrases(TUTOR_HINT_PATTERNS)
TUTOR_CHECK_RE = _compile_phrases(TUTOR_CHECK_PATTERNS)


BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SEG_WORDS = 200
SEG_STRIDE = 150
SEG_MIN_WORDS = 20


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(p in text for p in patterns)


def _tokens(text: str) -> list[str]:
    return [t.lower().replace(",", ".") for t in WORD_RE.findall(text)]


def _alpha_tokens(text: str) -> list[str]:
    return [t.lower() for t in ALPHA_RE.findall(text)]


def _content_words(text: str) -> list[str]:
    return [t for t in _alpha_tokens(text) if t not in STOPWORDS and len(t) > 2]


def _parse_timestamp(value: str) -> float | None:
    """Parse HH:MM:SS or an ISO-like datetime into seconds.

    Transcript timestamps observed in the competition are mostly HH:MM:SS.  The fallback
    keeps the extractor robust if the runtime data contains a full datetime string.
    """
    value = (value or "").strip()
    if not value:
        return None
    m = re.search(r"(?:^|T|\s)(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?$", value)
    if m:
        h, minute, sec = map(int, m.groups())
        return float(h * 3600 + minute * 60 + sec)
    try:
        ts = pd.to_datetime(value, errors="raise")
        return float(ts.value / 1e9)
    except Exception:
        return None


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0) and not pd.isna(b) else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _run_stats(roles: Sequence[str], target: str) -> tuple[float, int]:
    lengths: list[int] = []
    cur = 0
    for role in roles:
        if role == target:
            cur += 1
        elif cur:
            lengths.append(cur)
            cur = 0
    if cur:
        lengths.append(cur)
    return (_mean_or_zero(lengths), max(lengths) if lengths else 0)


@dataclass(slots=True)
class Turn:
    role: str
    content: str
    low: str
    words: list[str]
    alpha: list[str]
    n_words: int
    timestamp: float | None
    is_math: bool
    is_short_math: bool
    is_reason_math: bool
    is_explanation: bool
    is_self_correct: bool
    is_confusion: bool
    is_question: bool
    is_correction: bool
    is_praise: bool
    is_hint: bool
    is_check: bool
    is_backchannel: bool


def _turn(role: str, content: str, timestamp: str) -> Turn:
    role = (role or "").strip().lower()
    content = content or ""
    low = " ".join(content.lower().split())
    words = _tokens(content)
    alpha = [t for t in words if t and t[0].isalpha()]
    is_math = bool(NUMBER_RE.search(content)) or any(t in MATH_WORDS for t in alpha) \
        or any(t in {"+", "-", "*", "/", "=", "%", "<", ">"} for t in words)
    is_reason = bool(REASON_RE.search(low))
    is_explanation = is_reason or (is_math and len(words) >= 8)
    norm = re.sub(r"[^a-z0-9 ]+", " ", low)
    norm = " ".join(norm.split())
    return Turn(
        role=role,
        content=content,
        low=low,
        words=words,
        alpha=alpha,
        n_words=len(words),
        timestamp=_parse_timestamp(timestamp),
        is_math=is_math,
        is_short_math=is_math and len(words) <= 3,
        is_reason_math=is_math and is_reason,
        is_explanation=is_explanation,
        is_self_correct=bool(SELF_CORRECT_RE.search(low)),
        is_confusion=bool(CONFUSION_RE.search(low)),
        is_question="?" in content,
        is_correction=bool(TUTOR_CORRECTION_RE.search(low)),
        is_praise=bool(TUTOR_PRAISE_RE.search(low)),
        is_hint=bool(TUTOR_HINT_RE.search(low)),
        is_check=bool(TUTOR_CHECK_RE.search(low)),
        is_backchannel=norm in BACKCHANNELS,
    )


def read_turns(path: str | Path) -> list[Turn]:
    turns: list[Turn] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            turns.append(_turn(row.get("role") or "", row.get("content") or "",
                               row.get("timestamp") or ""))
    return turns


def extract_session_features(path: str | Path) -> dict[str, float | str]:
    """Extract one row of structured scalar features from one transcript CSV."""
    path = Path(path)
    turns = read_turns(path)
    out: dict[str, float | str] = {"session_id": path.stem}

    role_counts = Counter(t.role for t in turns)
    out.update(
        n_utt=len(turns),
        n_tutor=role_counts.get("tutor", 0),
        n_student=role_counts.get("student", 0),
        n_background=role_counts.get("background", 0),
        n_other=sum(v for k, v in role_counts.items() if k not in {"tutor", "student", "background"}),
    )

    for role in ("tutor", "student", "background"):
        out[f"w_{role}"] = sum(t.n_words for t in turns if t.role == role)
    out["w_total"] = sum(t.n_words for t in turns)

    ts = [t.timestamp for t in turns if t.timestamp is not None]
    out["n_ts"] = len(ts)
    out["n_ts_bad"] = len(turns) - len(ts)
    if ts:
        diffs = np.diff(np.asarray(ts, dtype=float))
        out["dur_s"] = max(0.0, float(ts[-1] - ts[0]))
        out["n_ts_decrease"] = int((diffs < 0).sum())
        out["n_ts_same_adj"] = int((diffs == 0).sum())
        positive = diffs[diffs >= 0]
        out["max_gap_s"] = float(positive.max()) if len(positive) else 0.0
        out["mean_gap_s"] = float(positive.mean()) if len(positive) else 0.0
    else:
        out.update(dur_s=np.nan, n_ts_decrease=0, n_ts_same_adj=0,
                   max_gap_s=0.0, mean_gap_s=0.0)

    out["tutor_q"] = sum(t.is_question for t in turns if t.role == "tutor")
    out["student_q"] = sum(t.is_question for t in turns if t.role == "student")
    out["background_q"] = sum(t.is_question for t in turns if t.role == "background")
    out["n_empty"] = sum(not t.words for t in turns)
    out["n_unclear"] = sum(t.low.count("[unclear]") for t in turns)
    out["n_bracket"] = sum(len(BRACKET_RE.findall(t.content)) for t in turns)

    tagged_rows = 0
    tags = 0
    contradictions = 0
    for t in turns:
        found = [x.lower() for x in SPEAKER_TAG_RE.findall(t.content)]
        if found:
            tagged_rows += 1
            tags += len(found)
            if t.role in {"tutor", "student", "background"} and any(x != t.role for x in found):
                contradictions += 1
    out["n_speaker_tag_rows"] = tagged_rows
    out["n_speaker_tags"] = tags
    out["n_role_contradict_rows"] = contradictions

    long_bg = [t for t in turns if t.role == "background" and t.n_words >= 15]
    out["n_long_background"] = len(long_bg)
    out["w_long_background"] = sum(t.n_words for t in long_bg)

    students = [t for t in turns if t.role == "student"]
    tutors = [t for t in turns if t.role == "tutor"]
    out.update(
        student_math_turns=sum(t.is_math for t in students),
        student_short_math_turns=sum(t.is_short_math for t in students),
        student_reasoning_math_turns=sum(t.is_reason_math for t in students),
        student_number_word_turns=sum(any(w in NUMBER_WORDS for w in t.alpha) for t in students),
        student_explanation_turns=sum(t.is_explanation for t in students),
        student_self_correct_turns=sum(t.is_self_correct for t in students),
        student_confusion_turns=sum(t.is_confusion for t in students),
        student_backchannel_turns=sum(t.is_backchannel for t in students),
        tutor_math_turns=sum(t.is_math for t in tutors),
        tutor_correction_turns=sum(t.is_correction for t in tutors),
        tutor_praise_turns=sum(t.is_praise for t in tutors),
        tutor_hint_turns=sum(t.is_hint for t in tutors),
        tutor_check_turns=sum(t.is_check for t in tutors),
        tutor_backchannel_turns=sum(t.is_backchannel for t in tutors),
    )

    # Consecutive-speaker / alternation features, ignoring background rows.
    dialogue_roles = [t.role for t in turns if t.role in {"tutor", "student"}]
    if len(dialogue_roles) >= 2:
        alt = sum(a != b for a, b in zip(dialogue_roles[:-1], dialogue_roles[1:]))
        out["alternation_rate"] = alt / (len(dialogue_roles) - 1)
    else:
        out["alternation_rate"] = 0.0
    out["same_role_rate"] = 1.0 - float(out["alternation_rate"])
    out["mean_student_run"], out["max_student_run"] = _run_stats(dialogue_roles, "student")
    out["mean_tutor_run"], out["max_tutor_run"] = _run_stats(dialogue_roles, "tutor")

    # Adjacent move-response transitions.  Background rows are skipped, but we do not jump
    # over another tutor/student contribution because that would make the relation ambiguous.
    dialogue = [t for t in turns if t.role in {"tutor", "student"}]
    after_q: list[int] = []
    after_corr: list[int] = []
    after_hint: list[int] = []
    reason_after_q = reason_after_corr = reason_after_hint = 0
    short_math_after_q = 0
    student_to_praise = 0
    for a, b in zip(dialogue[:-1], dialogue[1:]):
        if a.role == "tutor" and b.role == "student":
            if a.is_question:
                after_q.append(b.n_words)
                reason_after_q += int(b.is_explanation)
                short_math_after_q += int(b.is_short_math)
            if a.is_correction:
                after_corr.append(b.n_words)
                reason_after_corr += int(b.is_explanation)
            if a.is_hint:
                after_hint.append(b.n_words)
                reason_after_hint += int(b.is_explanation)
        elif a.role == "student" and b.role == "tutor" and b.is_praise:
            student_to_praise += 1
    out.update(
        n_tutorq_to_student=len(after_q),
        mean_student_words_after_tutorq=_mean_or_zero(after_q),
        n_reason_after_tutorq=reason_after_q,
        n_short_math_after_tutorq=short_math_after_q,
        n_correction_to_student=len(after_corr),
        mean_student_words_after_correction=_mean_or_zero(after_corr),
        n_reason_after_correction=reason_after_corr,
        n_hint_to_student=len(after_hint),
        mean_student_words_after_hint=_mean_or_zero(after_hint),
        n_reason_after_hint=reason_after_hint,
        n_student_to_praise=student_to_praise,
    )

    # First/middle/final thirds by utterance index.  Index-based thirds are deterministic
    # even when timestamps are missing or malformed.
    n = max(1, len(turns))
    for third_name, lo, hi in (("first3", 0.0, 1 / 3), ("mid3", 1 / 3, 2 / 3),
                               ("last3", 2 / 3, 1.000001)):
        part = [t for i, t in enumerate(turns) if lo <= i / n < hi and t.role == "student"]
        out[f"student_words_{third_name}"] = sum(t.n_words for t in part)
        out[f"student_turns_{third_name}"] = len(part)
        out[f"student_math_{third_name}"] = sum(t.is_math for t in part)
        out[f"student_short_math_{third_name}"] = sum(t.is_short_math for t in part)
        out[f"student_reason_{third_name}"] = sum(t.is_explanation for t in part)
        out[f"student_self_correct_{third_name}"] = sum(t.is_self_correct for t in part)

    return out


def add_derived_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable ratios and progress deltas to a raw session-feature frame."""
    df = df.copy()
    eps = 1.0

    def ratio(name: str, a: str, b: str, offset: float = eps):
        if a in df and b in df:
            df[name] = df[a] / (df[b] + offset)

    ratio("student_share_words_v2", "w_student", "w_total")
    ratio("tutor_share_words_v2", "w_tutor", "w_total")
    ratio("student_words_per_turn_v2", "w_student", "n_student")
    ratio("tutor_words_per_turn_v2", "w_tutor", "n_tutor")
    ratio("student_math_share", "student_math_turns", "n_student")
    ratio("student_short_math_share", "student_short_math_turns", "student_math_turns")
    ratio("student_short_math_per_word", "student_short_math_turns", "w_student")
    ratio("student_reasoning_math_share", "student_reasoning_math_turns", "student_math_turns")
    ratio("student_explanation_share", "student_explanation_turns", "n_student")
    ratio("student_self_correct_share", "student_self_correct_turns", "n_student")
    ratio("student_confusion_share", "student_confusion_turns", "n_student")
    ratio("student_backchannel_share", "student_backchannel_turns", "n_student")
    ratio("tutor_correction_per_student", "tutor_correction_turns", "n_student")
    ratio("tutor_praise_per_student", "tutor_praise_turns", "n_student")
    ratio("tutor_hint_per_student", "tutor_hint_turns", "n_student")
    ratio("tutor_check_per_student", "tutor_check_turns", "n_student")
    ratio("tutor_question_per_student", "tutor_q", "n_student")
    ratio("unclear_per_turn_v2", "n_unclear", "n_utt")
    ratio("speaker_tag_share", "n_speaker_tag_rows", "n_utt")
    ratio("role_contradiction_share", "n_role_contradict_rows", "n_utt")
    ratio("long_background_share", "n_long_background", "n_background")
    ratio("background_word_share", "w_background", "w_total")
    ratio("words_per_second_v2", "w_total", "dur_s")
    ratio("turns_per_minute", "n_utt", "dur_s", offset=1 / 60)
    if "turns_per_minute" in df:
        df["turns_per_minute"] *= 60.0

    if {"tutor_praise_turns", "tutor_correction_turns", "n_student"}.issubset(df.columns):
        df["praise_minus_correction_rate"] = (
            df["tutor_praise_turns"] - df["tutor_correction_turns"]
        ) / (df["n_student"] + 1.0)

    for prefix in ("student_words", "student_reason", "student_math",
                   "student_short_math", "student_self_correct"):
        a, b = f"{prefix}_first3", f"{prefix}_last3"
        if a in df and b in df:
            df[f"{prefix}_delta_last_first"] = df[b] - df[a]
            df[f"{prefix}_ratio_last_first"] = (df[b] + 1.0) / (df[a] + 1.0)

    # Within-third rates remove the trivial effect of unequal transcript length.
    for third in ("first3", "mid3", "last3"):
        turns = f"student_turns_{third}"
        words = f"student_words_{third}"
        if turns in df:
            for signal in ("student_reason", "student_math", "student_short_math",
                           "student_self_correct"):
                col = f"{signal}_{third}"
                if col in df:
                    df[f"{col}_share"] = df[col] / (df[turns] + 1.0)
        if turns in df and words in df:
            df[f"student_words_per_turn_{third}"] = df[words] / (df[turns] + 1.0)

    for count, total, out_name in (
        ("n_reason_after_tutorq", "n_tutorq_to_student", "reason_after_tutorq_share"),
        ("n_short_math_after_tutorq", "n_tutorq_to_student", "short_math_after_tutorq_share"),
        ("n_reason_after_correction", "n_correction_to_student", "reason_after_correction_share"),
        ("n_reason_after_hint", "n_hint_to_student", "reason_after_hint_share"),
    ):
        if count in df and total in df:
            df[out_name] = df[count] / (df[total] + 1.0)

    return df


def _extract_worker(path: str) -> dict[str, float | str]:
    return extract_session_features(path)


def build_session_feature_table(
    transcripts_dir: str | Path,
    session_ids: Iterable[str] | None = None,
    workers: int | None = None,
) -> pd.DataFrame:
    """Extract session features for a directory of transcript CSVs."""
    tdir = Path(transcripts_dir).expanduser()
    if session_ids is None:
        paths = sorted(tdir.glob("*.csv"))
    else:
        paths = [tdir / f"{sid}.csv" for sid in session_ids]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} transcript files; first: {missing[0]}")
    workers = workers or max(1, min(16, (os.cpu_count() or 4) - 1))
    if workers == 1:
        rows = [_extract_worker(str(p)) for p in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_extract_worker, map(str, paths), chunksize=32))
    df = add_derived_session_features(pd.DataFrame(rows))
    return df.sort_values("session_id").reset_index(drop=True)


def _objective_coverage_from_text(objective: str, text: str) -> dict[str, float]:
    lo_words = _content_words(objective)
    if not lo_words:
        return dict(overlap=np.nan, hit_count=0.0, mention_count=0.0,
                    bigram_overlap=np.nan, number_overlap=np.nan)
    text_alpha = _alpha_tokens(text)
    text_set = set(text_alpha)
    hits = sum(w in text_set for w in set(lo_words))
    mention_count = sum(Counter(text_alpha)[w] for w in set(lo_words))
    overlap = hits / len(set(lo_words))

    lo_bigrams = set(zip(lo_words[:-1], lo_words[1:]))
    text_bigrams = set(zip(text_alpha[:-1], text_alpha[1:]))
    bigram_overlap = (len(lo_bigrams & text_bigrams) / len(lo_bigrams)) if lo_bigrams else np.nan

    lo_nums = set(NUMBER_RE.findall(objective.lower())) | (set(lo_words) & NUMBER_WORDS)
    text_nums = set(NUMBER_RE.findall(text.lower())) | (text_set & NUMBER_WORDS)
    number_overlap = (len(lo_nums & text_nums) / len(lo_nums)) if lo_nums else np.nan
    return dict(overlap=float(overlap), hit_count=float(hits), mention_count=float(mention_count),
                bigram_overlap=float(bigram_overlap) if not pd.isna(bigram_overlap) else np.nan,
                number_overlap=float(number_overlap) if not pd.isna(number_overlap) else np.nan)


def _response_lexical_worker(payload: tuple[str, list[tuple[str, str]]]) -> list[dict[str, float | str]]:
    path, responses = payload
    turns = read_turns(path)
    text_by_role = {
        role: " ".join(t.content for t in turns if t.role == role)
        for role in ("student", "tutor", "background")
    }
    text_by_role["dialogue"] = " ".join(t.content for t in turns if t.role in {"student", "tutor"})
    rows: list[dict[str, float | str]] = []
    for response_id, objective in responses:
        row: dict[str, float | str] = {"response_id": response_id}
        for role, text in text_by_role.items():
            cov = _objective_coverage_from_text(objective, text)
            for key, value in cov.items():
                row[f"{role}_lo_{key}"] = value
        # Student-vs-tutor contrasts are often more useful than either raw coverage.
        row["student_minus_tutor_lo_overlap"] = (
            float(row["student_lo_overlap"]) - float(row["tutor_lo_overlap"])
        )
        row["student_over_dialogue_lo_overlap"] = _safe_div(
            float(row["student_lo_overlap"]), float(row["dialogue_lo_overlap"]) + 1e-3
        )
        rows.append(row)
    return rows


def build_response_lexical_features(
    train_features: pd.DataFrame,
    transcripts_dir: str | Path,
    workers: int | None = None,
) -> pd.DataFrame:
    """Build speaker-specific lexical objective coverage for each response."""
    required = {"response_id", "session_id", "learning_objective"}
    missing = required - set(train_features.columns)
    if missing:
        raise ValueError(f"train_features is missing columns: {sorted(missing)}")
    tdir = Path(transcripts_dir).expanduser()
    payloads: list[tuple[str, list[tuple[str, str]]]] = []
    for sid, g in train_features.groupby("session_id", sort=True):
        pairs = list(zip(g["response_id"].astype(str), g["learning_objective"].fillna("").astype(str)))
        payloads.append((str(tdir / f"{sid}.csv"), pairs))
    workers = workers or max(1, min(16, (os.cpu_count() or 4) - 1))
    if workers == 1:
        chunks = [_response_lexical_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            chunks = list(ex.map(_response_lexical_worker, payloads, chunksize=16))
    rows = [row for chunk in chunks for row in chunk]
    return pd.DataFrame(rows).sort_values("response_id").reset_index(drop=True)


def segment_role_text(turns: Sequence[Turn], role: str,
                      seg_words: int = SEG_WORDS, stride: int = SEG_STRIDE,
                      min_words: int = SEG_MIN_WORDS) -> list[str]:
    words: list[str] = []
    for t in turns:
        if t.role == role:
            words.extend(t.content.split())
    if not words:
        return []
    out: list[str] = []
    for start in range(0, max(1, len(words) - stride), stride):
        seg = words[start:start + seg_words]
        if len(seg) >= min_words or start == 0:
            out.append(" ".join(seg))
    return out


def build_student_bge_coverage(
    train_features: pd.DataFrame,
    transcripts_dir: str | Path,
    model_name_or_path: "str | object",
    batch_size: int = 256,
    session_batch_size: int = 96,
    device: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute objective-to-student-window BGE coverage per response.

    The implementation streams session batches so it does not retain all transcript segment
    embeddings in memory.  It uses the same query-prefix convention as ACE's existing BGE
    coverage pass.
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("student BGE coverage requires torch and sentence-transformers") from exc

    required = {"response_id", "session_id", "learning_objective"}
    missing = required - set(train_features.columns)
    if missing:
        raise ValueError(f"train_features is missing columns: {sorted(missing)}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Accept an ALREADY-LOADED encoder. The submission path has a SentenceTransformer
    # resident (main.py stage 3) and must not instantiate a second one: that would
    # double VRAM at the exact point the vLLM engine is about to start.
    if hasattr(model_name_or_path, "encode"):
        model = model_name_or_path
    else:
        model = SentenceTransformer(model_name_or_path, device=device)
    objective_texts = train_features["learning_objective"].fillna("").astype(str).drop_duplicates().tolist()
    emb_obj = model.encode(
        [BGE_QUERY_PREFIX + t for t in objective_texts],
        batch_size=min(batch_size, 128),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    obj_ix = {text: i for i, text in enumerate(objective_texts)}

    tdir = Path(transcripts_dir).expanduser()
    by_session = {sid: g.copy() for sid, g in train_features.groupby("session_id", sort=True)}
    session_ids = list(by_session)
    rows: list[dict[str, float | str]] = []
    for b0 in range(0, len(session_ids), session_batch_size):
        batch_sids = session_ids[b0:b0 + session_batch_size]
        segments: list[str] = []
        bounds: dict[str, tuple[int, int]] = {}
        for sid in batch_sids:
            turns = read_turns(tdir / f"{sid}.csv")
            segs = segment_role_text(turns, "student")
            bounds[sid] = (len(segments), len(segments) + len(segs))
            segments.extend(segs)
        if segments:
            emb_seg = model.encode(
                segments,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            emb_seg = np.empty((0, emb_obj.shape[1]), dtype=np.float32)

        for sid in batch_sids:
            a, b = bounds[sid]
            g = by_session[sid]
            for row in g.itertuples(index=False):
                out: dict[str, float | str] = {"response_id": str(row.response_id)}
                if b <= a:
                    out.update(
                        student_coverage_max=np.nan,
                        student_coverage_top3=np.nan,
                        student_coverage_top5=np.nan,
                        student_coverage_gap=np.nan,
                        student_coverage_std=np.nan,
                        student_coverage_n_ge_04=0.0,
                        student_coverage_n_ge_05=0.0,
                        student_coverage_n_ge_06=0.0,
                        student_coverage_best_pos=np.nan,
                    )
                else:
                    sims = emb_seg[a:b] @ emb_obj[obj_ix[str(row.learning_objective)]]
                    ordered = np.sort(sims)
                    k3, k5 = min(3, len(sims)), min(5, len(sims))
                    best = int(np.argmax(sims))
                    out.update(
                        student_coverage_max=float(sims[best]),
                        student_coverage_top3=float(ordered[-k3:].mean()),
                        student_coverage_top5=float(ordered[-k5:].mean()),
                        student_coverage_gap=float(sims[best] - ordered[-k3:].mean()),
                        student_coverage_std=float(sims.std()),
                        student_coverage_n_ge_04=float((sims >= 0.4).sum()),
                        student_coverage_n_ge_05=float((sims >= 0.5).sum()),
                        student_coverage_n_ge_06=float((sims >= 0.6).sum()),
                        student_coverage_best_pos=float(best / max(1, len(sims) - 1)),
                    )
                rows.append(out)
        if show_progress:
            # Gated with the tqdm bar: at inference this counts TEST sessions, and the
            # rules forbid emitting any aggregate of the test data. Offline builds keep it.
            print(f"[student-bge] sessions {min(b0 + session_batch_size, len(session_ids)):,}/"
                  f"{len(session_ids):,}", flush=True)

    return pd.DataFrame(rows).sort_values("response_id").reset_index(drop=True)


# Explicit lists used by the training script.  Raw columns already available in ACE's
# transcript_stats.csv are omitted where possible; the v2 model can concatenate its old
# feature list with these genuinely new fields.
TABULAR_V2_SESSION_FEATURES = [
    "w_background", "n_other", "n_ts", "n_ts_bad", "n_ts_decrease", "n_ts_same_adj",
    "max_gap_s", "mean_gap_s", "background_q", "n_empty", "n_speaker_tag_rows",
    "n_speaker_tags", "n_role_contradict_rows", "n_long_background", "w_long_background",
    "student_math_turns", "student_short_math_turns", "student_reasoning_math_turns",
    "student_number_word_turns", "student_explanation_turns", "student_self_correct_turns",
    "student_confusion_turns", "student_backchannel_turns", "tutor_math_turns",
    "tutor_correction_turns", "tutor_praise_turns", "tutor_hint_turns", "tutor_check_turns",
    "tutor_backchannel_turns", "alternation_rate", "same_role_rate", "mean_student_run",
    "max_student_run", "mean_tutor_run", "max_tutor_run", "n_tutorq_to_student",
    "mean_student_words_after_tutorq", "n_reason_after_tutorq", "n_short_math_after_tutorq",
    "n_correction_to_student", "mean_student_words_after_correction",
    "n_reason_after_correction", "n_hint_to_student", "mean_student_words_after_hint",
    "n_reason_after_hint", "n_student_to_praise", "student_words_first3",
    "student_words_mid3", "student_words_last3", "student_turns_first3", "student_turns_mid3",
    "student_turns_last3", "student_math_first3", "student_math_mid3", "student_math_last3",
    "student_short_math_first3", "student_short_math_mid3", "student_short_math_last3",
    "student_reason_first3", "student_reason_mid3", "student_reason_last3",
    "student_self_correct_first3", "student_self_correct_mid3", "student_self_correct_last3",
    "student_share_words_v2", "tutor_share_words_v2", "student_words_per_turn_v2",
    "tutor_words_per_turn_v2", "student_math_share", "student_short_math_share",
    "student_short_math_per_word", "student_reasoning_math_share", "student_explanation_share",
    "student_self_correct_share", "student_confusion_share", "student_backchannel_share",
    "tutor_correction_per_student", "tutor_praise_per_student", "tutor_hint_per_student",
    "tutor_check_per_student", "tutor_question_per_student", "unclear_per_turn_v2",
    "speaker_tag_share", "role_contradiction_share", "long_background_share",
    "background_word_share", "words_per_second_v2", "turns_per_minute",
    "praise_minus_correction_rate", "student_words_delta_last_first",
    "student_words_ratio_last_first", "student_reason_delta_last_first",
    "student_reason_ratio_last_first", "student_math_delta_last_first",
    "student_math_ratio_last_first", "student_short_math_delta_last_first",
    "student_short_math_ratio_last_first", "student_self_correct_delta_last_first",
    "student_self_correct_ratio_last_first", "student_reason_first3_share",
    "student_reason_mid3_share", "student_reason_last3_share", "student_math_first3_share",
    "student_math_mid3_share", "student_math_last3_share", "student_short_math_first3_share",
    "student_short_math_mid3_share", "student_short_math_last3_share",
    "student_self_correct_first3_share", "student_self_correct_mid3_share",
    "student_self_correct_last3_share", "student_words_per_turn_first3",
    "student_words_per_turn_mid3", "student_words_per_turn_last3",
    "reason_after_tutorq_share", "short_math_after_tutorq_share",
    "reason_after_correction_share", "reason_after_hint_share",
]

TABULAR_V2_RESPONSE_LEXICAL_FEATURES = [
    f"{role}_lo_{metric}"
    for role in ("student", "tutor", "background", "dialogue")
    for metric in ("overlap", "hit_count", "mention_count", "bigram_overlap", "number_overlap")
] + ["student_minus_tutor_lo_overlap", "student_over_dialogue_lo_overlap"]

TABULAR_V2_BGE_FEATURES = [
    "student_coverage_max", "student_coverage_top3", "student_coverage_top5",
    "student_coverage_gap", "student_coverage_std", "student_coverage_n_ge_04",
    "student_coverage_n_ge_05", "student_coverage_n_ge_06", "student_coverage_best_pos",
]
