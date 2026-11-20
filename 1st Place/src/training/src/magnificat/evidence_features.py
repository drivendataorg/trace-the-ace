"""Evidence features: the performance-record, contingency and error-repair blocks.

One definition for both the trainer (`src/training/scripts/build_evidence_features.py`)
and `models/submission/main.py`, as with `model_e2.py`: training and inference must build
identical columns, or the feature matrix drifts between them.

Three blocks, all computable from a transcript alone at inference time.

  R (`recdst_*`), the performance record: a per-turn right/wrong tally. The per-turn
      correctness label comes from a distilled annotator
      (`turn_correctness_clf.joblib`, TF-IDF + logistic regression,
      7.3 MB, CPU) trained on the Qwen3.5-122B annotation; the 122B itself cannot run in
      the 6 h offline container.

  C (`ctg_*`), contingency in the correct order: each tutor move is tagged with the
      student state at the moment it occurs (tell-at-impasse vs tell-when-fluent).

  P (`rep_*`), the error-repair arc: after a tutor flags an error, which strategy follows
      (prompting, explaining or giving the answer), whether the next graded answer
      recovers, and whether the error recurs.

Turn structure mirrors dialogue-kt's `process_dialogue` exactly (consecutive same-role
utterances merged; a turn = teacher block + student block), so the distilled annotator
sees the same unit it was trained on.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .dialogue_features import CONFUSION, RESOLUTION, ELICIT, TELL, CORRECTIVE
from .features import (
    _FEEDBACK_INCORRECT, _PROMPT_SELF_CORRECTION, _PROMPT_EXPLANATION,
    _EXPLAINING_PROCEDURAL, _EXPLAINING_CONCEPTUAL, _GIVING_ANSWER,
    _GIVING_ANSWER_BARE, _BACKCHANNEL,
)

K_EXPIRE = 10          # impasse locality (VanLehn)
REPAIR_WINDOW = 6      # utterances after a corrective in which a re-error still counts
VERDICT_HEAD = 90      # a verdict comes at the start of the reply, not later in it

# Verdict lexicons for the regex answer log. Not `CONFIRM`/`CORRECTIVE` from
# dialogue_features: those count how often a tutor confirms, and `CONFIRM` includes
# "good"/"great"/"nice", which fire constantly as social encouragement. The record block
# uses the distilled annotator; the regex verdicts feed only the repair-arc recovery
# signal.
VERDICT_POS = re.compile(
    r"\b(exactly|that'?s? (right|correct|it)|that is (right|correct)|"
    r"well done|good job|perfect|brilliant|spot on|fantastic|superb|lovely|"
    r"you'?(ve)? got it|nailed it|bang on|absolutely right|"
    r"correct\b|yes[,.!]? (that|it|you)('?s| is| are| do| did)?\b)",
    flags=re.IGNORECASE)
VERDICT_NEG = re.compile(
    r"\b(not quite|not right|not correct|not exactly|that'?s? (wrong|incorrect)|"
    r"that is (wrong|incorrect)|incorrect\b|try again|have another (look|go|try)|"
    r"are you sure|double.?check|check (that|your|it) again|look at that again|"
    r"close but|almost\b|nearly\b|not what|"
    r"no[,.!]? (that|it|not|the answer|we|you)\b|nope\b|"
    r"careful\b|watch out|hold on\b|hang on\b)",
    flags=re.IGNORECASE)

RECORD_STATS = ["accuracy", "last_1", "last_3", "last_5", "recency_acc",
                "run_correct_share", "run_incorrect_share", "half_delta",
                "trend", "pos_last_incorrect", "early_acc", "late_acc"]
RECORD_COLS = RECORD_STATS + ["answered_share", "log_answered"]
CTG_COLS = ["ctg_tell_imp", "ctg_tell_flu", "ctg_elicit_imp", "ctg_elicit_flu",
            "ctg_tell_contrast", "ctg_elicit_contrast", "ctg_imp_share",
            "ctg_imp_utt_share", "ctg_late_imp_share", "ctg_episodes_per_utt"]
REP_COLS = ["rep_prompt_share", "rep_explain_share", "rep_answer_share",
            "rep_recovery_rate", "rep_re_error_rate", "rep_arcs_per_tutor_turn",
            "rep_has_arc"]
#: the evidence block, in a fixed order (train/inference must agree)
EVIDENCE_COLS = [f"recdst_{c}" for c in RECORD_COLS] + CTG_COLS + REP_COLS


# --------------------------------------------------------------------------- turns
def build_turns(tdf: pd.DataFrame):
    """Replicate dialogue-kt's `process_dialogue` on a raw transcript.

    ``background`` rows and empty utterances are dropped, matching
    `build_traceace_kt.build_session_dialogue`.

    Args:
        tdf: one session's transcript (``TranscriptStore`` format).

    Returns:
        ``(turns, utts)``: ``turns`` is the annotator's indexing unit, a list of
        ``{"turn": i, "teacher": str, "student": str}``; ``utts`` is the flat
        ``[(role, lowercased_text), ...]`` stream the state machines walk, with role
        ``"teacher"`` or ``"student"``.
    """
    if "utterance_id" in tdf.columns:
        tdf = tdf.sort_values("utterance_id")
    utts = []
    for role, content in zip(tdf["role"], tdf["content"]):
        role = str(role).strip().lower()
        content = "" if pd.isna(content) else str(content).strip()
        if not content or role not in ("tutor", "student"):
            continue
        utts.append(("teacher" if role == "tutor" else "student", content))
    if not utts:
        return [], []

    cur_role = utts[0][0]
    cur = {"turn": 0 if cur_role == "student" else 1, "teacher": "", "student": ""}
    out = []
    for role, content in utts:
        if role == "teacher" and cur_role == "student":
            out.append(cur)
            cur = {"turn": cur["turn"] + 1, "teacher": "", "student": ""}
        cur_role = role
        cur[role] = (cur[role] + " " + content).strip() if cur[role] else content
    if cur["student"]:
        out.append(cur)
    return out, [(r, t.lower()) for r, t in utts]


def turn_text(turns, i):
    """Build the distilled annotator's input for turn ``i``.

    The question it grades against, the answer, and the tutor's reaction: the same three
    fields and truncations as the training table in `build_evidence_features.py`.

    Args:
        turns: the turn list from :func:`build_turns`.
        i: index of the turn.

    Returns:
        ``"<tutor[-220:]> [A] <student[:220]> [R] <next tutor[:160]>"``.
    """
    nxt = turns[i + 1]["teacher"] if i + 1 < len(turns) else ""
    return (turns[i]["teacher"][-220:] + " [A] " + turns[i]["student"][:220]
            + " [R] " + nxt[:160])


def tutor_verdict(text: str):
    """Read a tutor block as a retrospective verdict on the student's previous turn.

    Only the first `VERDICT_HEAD` characters are searched (a verdict is the reply's
    opening move). Negative wins ties, so "not quite, good try" is a correction; a bare
    backchannel is never a positive verdict.

    Args:
        text: the tutor block that follows the student's turn.

    Returns:
        1 (correct), 0 (incorrect) or None (no verdict).
    """
    if not text:
        return None
    head = text[:VERDICT_HEAD]
    if VERDICT_NEG.search(head) or _FEEDBACK_INCORRECT.search(head):
        return 0
    if VERDICT_POS.search(head) and not _BACKCHANNEL.match(text):
        return 1
    return None


# ------------------------------------------------------------------ record block
def record_block(seq, pos, n_turns, prefix):
    """Compute rate-based performance-record features from a session's answer log.

    Run length and last-incorrect binarise at 0.5; everything else uses the raw value, so
    one code path serves hard and soft logs.

    Rates and shares plus one log-count rather than raw counts, which confound with
    transcript length (``model_e2`` drops raw counts too). There is both an early window
    (answers in the first 30% of the session's turns) and a late one (the last 30%).

    Args:
        seq: per-answer correctness, binary (a hard answer log) or soft probabilities in
            [0, 1] (the distilled annotator), in session order.
        pos: turn index of each answer.
        n_turns: number of turns in the session.
        prefix: column prefix, e.g. ``"recdst"``.

    Returns:
        Dict of ``{prefix}_{c}`` for ``RECORD_COLS``; the ``RECORD_STATS`` are NaN and the
        two counts 0 for an empty log.
    """
    a = np.asarray(seq, float)
    n = len(a)
    k = f"{prefix}_"
    if n == 0:
        return {k + c: np.nan for c in RECORD_STATS} | {
            k + "answered_share": 0.0, k + "log_answered": 0.0}

    b = (a >= 0.5).astype(float)
    w = np.exp(-(n - 1 - np.arange(n)) / 5.0)          # half-life ~3.5 answers
    runs_c = runs_i = cur_c = cur_i = 0
    for v in b:
        cur_c, cur_i = (cur_c + 1, 0) if v else (0, cur_i + 1)
        runs_c, runs_i = max(runs_c, cur_c), max(runs_i, cur_i)
    h = n // 2
    first = a[:h].mean() if h else np.nan
    second = a[h:].mean() if n - h else np.nan
    rel = np.asarray(pos, float) / max(n_turns, 1)
    early_mask, late_mask = rel <= 0.3, rel >= 0.7
    return {
        k + "accuracy": float(a.mean()),
        k + "last_1": float(a[-1]),
        k + "last_3": float(a[-3:].mean()),
        k + "last_5": float(a[-5:].mean()),
        k + "recency_acc": float((w * a).sum() / w.sum()),
        k + "run_correct_share": runs_c / n,
        k + "run_incorrect_share": runs_i / n,
        k + "half_delta": float(second - first) if (h and n - h) else np.nan,
        k + "trend": float(np.polyfit(np.arange(n), a, 1)[0]) if n > 2 else 0.0,
        k + "pos_last_incorrect": float(np.where(b == 0)[0][-1] / n) if (b == 0).any() else 1.0,
        k + "early_acc": float(a[early_mask].mean()) if early_mask.any() else np.nan,
        k + "late_acc": float(a[late_mask].mean()) if late_mask.any() else np.nan,
        k + "answered_share": n / max(n_turns, 1),
        k + "log_answered": float(np.log1p(n)),
    }


# ------------------------------------------------------------- contingency block
def contingency_block(utts):
    """Compute order-correct state × move rates.

    The state is impasse from a student confusion marker until a student resolution
    marker or a K=10-utterance expiry; each tutor move is tagged with the state at the
    moment it occurs. Adds the two impasse-minus-fluent contrast terms and a late-impasse
    share.

    Args:
        utts: the flat ``[(role, lowercased_text), ...]`` stream from :func:`build_turns`.

    Returns:
        Dict of the ten :data:`CTG_COLS` values.
    """
    n = len(utts)
    imp_until = -1
    c = dict(tell_imp=0, tell_flu=0, elicit_imp=0, elicit_flu=0, tut_imp=0, tut_flu=0)
    episodes = imp_utts = late_imp = 0
    late_start = 0.8 * n
    for i, (role, low) in enumerate(utts):
        state_imp = i <= imp_until
        if state_imp:
            imp_utts += 1
            if i >= late_start:
                late_imp += 1
        if role == "teacher":
            tag = "imp" if state_imp else "flu"
            c[f"tut_{tag}"] += 1
            if TELL.search(low):
                c[f"tell_{tag}"] += 1
            if ELICIT.search(low):
                c[f"elicit_{tag}"] += 1
        else:
            if CONFUSION.search(low):
                if not state_imp:
                    episodes += 1
                imp_until = max(imp_until, i + K_EXPIRE)
            elif RESOLUTION.search(low):
                imp_until = i - 1
    out = {}
    for mv in ("tell", "elicit"):
        for st in ("imp", "flu"):
            out[f"ctg_{mv}_{st}"] = c[f"{mv}_{st}"] / max(1, c[f"tut_{st}"])
    # give the trees the contrasts directly rather than relying on the booster
    # to reconstruct a difference of two rates
    out["ctg_tell_contrast"] = out["ctg_tell_imp"] - out["ctg_tell_flu"]
    out["ctg_elicit_contrast"] = out["ctg_elicit_imp"] - out["ctg_elicit_flu"]
    out["ctg_imp_share"] = c["tut_imp"] / max(1, c["tut_imp"] + c["tut_flu"])
    out["ctg_imp_utt_share"] = imp_utts / max(1, n)
    out["ctg_late_imp_share"] = late_imp / max(1, n - int(late_start))
    out["ctg_episodes_per_utt"] = episodes / max(1, n)
    return out


# ------------------------------------------------------------- repair-arc block
def _repair_kind(low: str):
    """Classify a tutor utterance's repair strategy, most specific pattern first.

    Args:
        low: lower-cased tutor utterance.

    Returns:
        ``"answer"``, ``"prompt"``, ``"explain"`` or None.
    """
    if _GIVING_ANSWER.search(low) or _GIVING_ANSWER_BARE.match(low.strip()):
        return "answer"
    if _PROMPT_SELF_CORRECTION.search(low) or _PROMPT_EXPLANATION.search(low) or ELICIT.search(low):
        return "prompt"
    if _EXPLAINING_PROCEDURAL.search(low) or _EXPLAINING_CONCEPTUAL.search(low) or TELL.search(low):
        return "explain"
    return None


def repair_block(utts, verdict_at):
    """Compute error-repair arc features; each arc starts at a tutor corrective.

    Per arc: which strategy the tutor's next substantive utterance takes, whether the next
    graded student answer recovers, and whether another corrective lands inside
    REPAIR_WINDOW (a re-error).

    Args:
        utts: the flat ``[(role, lowercased_text), ...]`` stream from :func:`build_turns`.
        verdict_at: flat utterance index -> 1/0 from :func:`verdict_index`, so recovery is
            measured on evidence rather than on the absence of a marker.

    Returns:
        Dict of the seven :data:`REP_COLS` values; the five rates are NaN when the session
        has no arc.
    """
    n = len(utts)
    kinds = {"prompt": 0, "explain": 0, "answer": 0}
    n_arcs = recovered = re_error = 0
    n_tutor = sum(1 for r, _ in utts if r == "teacher")
    for i, (role, low) in enumerate(utts):
        if role != "teacher" or not (CORRECTIVE.search(low) or _FEEDBACK_INCORRECT.search(low)):
            continue
        n_arcs += 1
        kind = _repair_kind(low)
        if kind is None:
            for j in range(i + 1, min(n, i + REPAIR_WINDOW)):
                if utts[j][0] == "teacher":
                    kind = _repair_kind(utts[j][1])
                    break
        if kind:
            kinds[kind] += 1
        for j in range(i + 1, n):
            v = verdict_at.get(j)
            if v is not None:
                recovered += v
                break
        for j in range(i + 1, min(n, i + 1 + REPAIR_WINDOW)):
            if utts[j][0] == "teacher" and (CORRECTIVE.search(utts[j][1])
                                            or _FEEDBACK_INCORRECT.search(utts[j][1])):
                re_error += 1
                break
    if n_arcs == 0:
        return {c: np.nan for c in REP_COLS[:5]} | {
            "rep_arcs_per_tutor_turn": 0.0, "rep_has_arc": 0.0}
    tot = max(1, sum(kinds.values()))
    return {
        "rep_prompt_share": kinds["prompt"] / tot,
        "rep_explain_share": kinds["explain"] / tot,
        "rep_answer_share": kinds["answer"] / tot,
        "rep_recovery_rate": recovered / n_arcs,
        "rep_re_error_rate": re_error / n_arcs,
        "rep_arcs_per_tutor_turn": n_arcs / max(1, n_tutor),
        "rep_has_arc": 1.0,
    }


def verdict_index(turns, utts):
    """Project the regex tutor verdicts onto flat utterance indices.

    The repair block asks "did the next graded answer recover", keyed by utterance, not
    turn.

    Args:
        turns: the turn list from :func:`build_turns`.
        utts: the flat utterance stream from :func:`build_turns`.

    Returns:
        Dict mapping the index of each turn's last student utterance to the 1/0 verdict of
        the following tutor block (turns with no verdict are omitted).
    """
    last_student_utt, cur_role, t_idx = {}, (utts[0][0] if utts else None), (
        turns[0]["turn"] if turns else 0)
    for j, (role, _) in enumerate(utts):
        if role == "teacher" and cur_role == "student":
            t_idx += 1
        cur_role = role
        if role == "student":
            last_student_utt[t_idx] = j
    out = {}
    for i, t in enumerate(turns):
        if not t["student"].strip():
            continue
        nxt = turns[i + 1]["teacher"] if i + 1 < len(turns) else ""
        v = tutor_verdict(nxt)
        if v is not None:
            j = last_student_utt.get(t["turn"])
            if j is not None:
                out[j] = v
    return out


# --------------------------------------------------------------------- entry point
def build_evidence_matrix(feats: pd.DataFrame, store=None, clf=None,
                          batch: int = 200_000, show_progress: bool = False) -> pd.DataFrame:
    """Build :data:`EVIDENCE_COLS` live from transcripts, keyed by ``response_id``.

    Follows `dialogue_features.build_dialogue_matrix`'s contract, so the inference path
    in `models/submission/main.py` reads the same way as every other block. Turn texts are
    vectorised in batches: a full corpus is ~2.4M student turns, and scoring them one
    session at a time would dominate the runtime budget. Each session's features depend
    only on that session (no pooling across samples).

    Args:
        feats: rows with ``response_id`` and ``session_id``.
        store: a ``TranscriptStore``; defaults to the training store.
        clf: the distilled annotator dict ``{"p_ans": pipeline, "p_cor": pipeline}``
            (``artifacts/turn_correctness_clf.joblib``). If None, the ``recdst_*`` columns
            come back all-NaN. That is a valid state: the ablation-augmented trainer sets
            the whole block to NaN in its ablation copies, so the booster has seen this case.
        batch: number of turn texts scored per call.
        show_progress: show a tqdm bar if tqdm is installed.

    Returns:
        DataFrame indexed by ``response_id`` with :data:`EVIDENCE_COLS` (float).
    """
    from . import data as _data

    store = store or _data.default_store()
    uniq = list(dict.fromkeys(feats["session_id"]))
    it = store.iter(uniq)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(uniq), desc="evidence")
        except ImportError:
            pass

    sess, texts, owner = {}, [], []
    for sid, tdf in it:
        try:
            turns, utts = build_turns(tdf)
        except Exception:
            turns, utts = [], []
        if not turns:
            sess[sid] = {c: np.nan for c in EVIDENCE_COLS}
            continue
        row = contingency_block(utts) | repair_block(utts, verdict_index(turns, utts))
        sess[sid] = row
        for i, t in enumerate(turns):
            if t["student"].strip():
                texts.append(turn_text(turns, i))
                owner.append((sid, i, len(turns)))

    # ---- distilled answer log, batched over the whole corpus -------------------
    logs: dict[str, tuple[list, list, int]] = {}
    if clf is not None and texts:
        arr = np.asarray(texts, dtype=object)
        p_ans = np.concatenate([clf["p_ans"].predict_proba(arr[i:i + batch])[:, 1]
                                for i in range(0, len(arr), batch)])
        p_cor = np.concatenate([clf["p_cor"].predict_proba(arr[i:i + batch])[:, 1]
                                for i in range(0, len(arr), batch)])
        for (sid, i, nt), pa, pc in zip(owner, p_ans, p_cor):
            if pa >= 0.5:
                s = logs.setdefault(sid, ([], [], nt))
                s[0].append(float(pc))
                s[1].append(i)

    for sid, row in sess.items():
        seq, pos, nt = logs.get(sid, ([], [], 1))
        row.update(record_block(seq, pos, nt, "recdst"))

    out = pd.DataFrame(
        [sess.get(s, {}) for s in feats["session_id"]],
        index=pd.Index(feats["response_id"], name="response_id"),
    )
    for c in EVIDENCE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[EVIDENCE_COLS].astype(float)
