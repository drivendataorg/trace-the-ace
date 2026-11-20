#!/usr/bin/env python
"""Build the evidence feature blocks (answer record, contingency, repair arcs) and turn_text.

Model E2's 70 columns are rates of surface markers: how often someone used a confusion, elicit
or confirm word. These blocks add three things E2 does not compute:

  R. Performance record. E2 has no column that says whether the student answered anything
     correctly during the session.
       - `rec122_*`: from the Qwen3.5-122B per-turn correctness annotation. It exists for
         training rows only, so it cannot be computed at inference.
       - `recprx_*`: from a transcript-only tutor-verdict proxy (below), which can be
         computed at inference inside the offline container.
       - `recdst_*`, the record block the shipped XGBoost model uses, is built from
         `turn_text.pkl.gz` by `distil_turn_correctness.py`.

  C. Contingency (student state x tutor move). A tutor move has no fixed valence; the fit
     between the move and the student's state does (telling at an impasse vs telling when
     the student is fluent). E2 has `tutor_tell_rate` and `confusion_rate` as two separate
     session-level scalars, from which the order-aware interaction cannot be recovered.

  P. Error-repair arc. After a tutor flags an error: which strategy follows (prompting
     self-correction, explaining or giving the answer) and whether the student recovers.
     E2 has the move rates but never conditions them on following an error.

Tutor-verdict proxy. A 122B annotator cannot run inside the 6 h offline container, so this
answer log comes from regexes. The tutor's retrospective verdict is the diagnostic signal, so
student turn t is labelled from the tutor's next utterance block: corrective or negative
feedback -> incorrect, confirmation or praise -> correct, neither -> unlabelled. The repair-arc
recovery rate reads the same verdicts. The script reports the proxy's agreement with the 122B
labels (accuracy, kappa, coverage).

The turn structure follows dialogue-kt's `process_dialogue` (consecutive same-role utterances
merged; a turn is a tutor block plus a student block), so the proxy and the 122B labels share
one turn index and can be compared row by row.

The script also writes the per-turn text table `turn_text.pkl.gz` (one row per student turn:
the preceding tutor text, the student text and the tutor's reply, truncated to 220/220/160
characters), the input of every distiller.

Training data only: no test data is read. Everything except `rec122_*` is regex over the
transcript and so can be computed at inference.

In:  data/train_transcripts/*.csv
     external/dialogue-kt/results/traceace_turn_labels_qwen35_122b.csv
Out: artifacts/evidence_features.csv (one row per session_id)
     artifacts/turn_text.pkl.gz
     figures/data/evidence_proxy_validation.md

Usage (reproduce.py step 01b, from build/workspace/; ~25 min):
  python scripts/build_evidence_features.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

#: Per-turn text truncation baked into `turn_text.pkl.gz`, set from --trunc in main().
#: These defaults are the shipped values. `nto_features`, `student_features` and `v2_features`
#: reproduce the same composition at inference, so changing one side without the other is a
#: train/serve skew that no in-domain experiment can detect.
_TT, _TS, _TR = 220, 220, 160
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from magnificat.dialogue_features import (  # noqa: E402
    CONFUSION, RESOLUTION, ELICIT, TELL, CORRECTIVE, CONFIRM,
)
from magnificat.features import (  # noqa: E402
    _FEEDBACK_INCORRECT, _PROMPT_SELF_CORRECTION, _PROMPT_EXPLANATION,
    _EXPLAINING_PROCEDURAL, _EXPLAINING_CONCEPTUAL, _GIVING_ANSWER,
    _GIVING_ANSWER_BARE, _BACKCHANNEL,
)

TX = ROOT / "data/train_transcripts"
LABELS_122B = ROOT / "external/dialogue-kt/results/traceace_turn_labels_qwen35_122b.csv"
K_EXPIRE = 10          # impasse expiry in utterances
REPAIR_WINDOW = 6      # utterances after a corrective in which a re-error still counts
VERDICT_HEAD = 90      # characters searched: a verdict opens the reply

# Verdict-specific lexicons, not `CONFIRM`/`CORRECTIVE` from dialogue_features: those are
# tuned to count how often a tutor confirms, and `CONFIRM` includes "good"/"great"/"nice",
# which fire constantly as social filler. A verdict has to be an assessment, so the
# positive list keeps only unambiguous assessments and the negative list adds the forms
# `CORRECTIVE` misses.
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


# --------------------------------------------------------------------------- turns
def build_turns(tdf: pd.DataFrame):
    """Split a raw transcript into turns the way dialogue-kt's process_dialogue does.

    `background` rows are dropped, as in build_traceace_kt.build_session_dialogue.

    Args:
        tdf: one transcript (utterance_id, role, content).

    Returns:
        ``(turns, utts)``: ``turns`` is a list of ``{"turn": i, "teacher": str,
        "student": str}`` (the turn index of the 122B labels); ``utts`` is the flat list of
        ``(role, lower-cased text)`` with role "teacher" or "student", for the state
        machines. Both are empty when the transcript has no tutor or student text.
    """
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


# ------------------------------------------------------------- the tutor verdict
def tutor_verdict(text: str):
    """Read a tutor utterance block as a verdict on the student's previous turn.

    Only the first `VERDICT_HEAD` characters are searched: a verdict is the reply's opening
    move, and searching the whole block would make a later "good" in "that's a good
    question" read as an assessment. Negative is checked first, so "not quite, good try" is
    a correction. A backchannel (`_BACKCHANNEL`, e.g. a bare "okay") is never a positive
    verdict: it is turn-taking, not assessment.

    Args:
        text: the tutor's utterance block.

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


def proxy_sequence(turns):
    """Build the proxy answer log: student turn t is graded by the tutor block of turn t+1.

    Args:
        turns: turns from `build_turns`.

    Returns:
        ``(seq, pos)``: the 0/1 verdicts and the index into ``turns`` of each graded turn.
    """
    seq, pos = [], []
    for i, t in enumerate(turns):
        if not t["student"].strip():
            continue
        nxt = turns[i + 1]["teacher"] if i + 1 < len(turns) else ""
        v = tutor_verdict(nxt)
        if v is not None:
            seq.append(v)
            pos.append(i)
    return seq, pos


# ------------------------------------------------------------- record features
RECORD_COLS = ["accuracy", "last_1", "last_3", "last_5", "recency_acc",
               "run_correct_share", "run_incorrect_share", "half_delta",
               "trend", "pos_last_incorrect", "early_acc", "late_acc"]


def record_block(seq, pos, n_turns, prefix):
    """Performance-record features of one session, as rates and shares.

    The run-length and last-incorrect features binarise at 0.5 and everything else uses the
    raw value, so one code path serves binary logs (122B labels, regex verdicts) and soft
    probabilities (the distilled annotator).

    Raw counts (`n_answered`, `n_correct`, `n_turns`) are avoided because they confound with
    transcript length. The same information is expressed here as shares and rates plus one
    log count, with early- and late-window accuracies added.

    Args:
        seq: answer log, 0/1 or probabilities in [0, 1], in session order.
        pos: turn index of each entry of ``seq``.
        n_turns: number of turns in the session.
        prefix: column prefix (e.g. "rec122", "recprx").

    Returns:
        Dict of ``{prefix}_<feature>`` values (the RECORD_COLS plus answered_share and
        log_answered); RECORD_COLS are NaN when the log is empty.
    """
    a = np.asarray(seq, float)
    n = len(a)
    k = f"{prefix}_"
    if n == 0:
        return {k + c: np.nan for c in RECORD_COLS} | {
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
    # Early and late windows by turn position: the first and the last 30% of the turns.
    rel = np.asarray(pos, float) / max(n_turns, 1)
    early_mask = rel <= 0.3
    late_mask = rel >= 0.7
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


# ------------------------------------------------------- contingency (block C)
def contingency_block(utts):
    """Order-aware student-state x tutor-move rates (block C, the `ctg_` columns).

    Per-state tell and elicit rates, two contrast terms and late-impasse shares. The state
    is "impasse" from a student confusion marker until a student resolution marker or
    K_EXPIRE utterances later; each tutor move is tagged with the state in force when it
    occurs.

    Args:
        utts: flat ``(role, lower-cased text)`` list from `build_turns`.

    Returns:
        Dict of the 10 ``ctg_*`` features.
    """
    n = len(utts)
    imp_until = -1
    c = dict(tell_imp=0, tell_flu=0, elicit_imp=0, elicit_flu=0, tut_imp=0, tut_flu=0)
    episodes = 0
    imp_utts = 0
    late_imp = 0
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
    # Give the booster the contrasts directly rather than relying on depth-7 trees to
    # reconstruct a difference of two rates.
    out["ctg_tell_contrast"] = out["ctg_tell_imp"] - out["ctg_tell_flu"]
    out["ctg_elicit_contrast"] = out["ctg_elicit_imp"] - out["ctg_elicit_flu"]
    out["ctg_imp_share"] = c["tut_imp"] / max(1, c["tut_imp"] + c["tut_flu"])
    out["ctg_imp_utt_share"] = imp_utts / max(1, n)
    out["ctg_late_imp_share"] = late_imp / max(1, n - int(late_start))
    out["ctg_episodes_per_utt"] = episodes / max(1, n)
    return out


# ------------------------------------------------------- repair arc (block P)
def _repair_kind(low: str, raw: str):
    """Classify a tutor utterance's repair strategy as "answer", "prompt", "explain" or None.

    Checked most specific first: giving the answer, then prompting, then explaining.
    """
    if _GIVING_ANSWER.search(low) or _GIVING_ANSWER_BARE.match(raw.strip()):
        return "answer"
    if _PROMPT_SELF_CORRECTION.search(low) or _PROMPT_EXPLANATION.search(low) or ELICIT.search(low):
        return "prompt"
    if _EXPLAINING_PROCEDURAL.search(low) or _EXPLAINING_CONCEPTUAL.search(low) or TELL.search(low):
        return "explain"
    return None


def repair_block(utts, verdict_at):
    """Error-repair arc features (block P, the `rep_` columns).

    An arc starts at a tutor corrective. For each one: which strategy the tutor's next
    substantive utterance takes, whether the student's next graded answer is correct, and
    whether another corrective follows within REPAIR_WINDOW utterances (the re-error or
    shallow-repair pattern).

    Args:
        utts: flat ``(role, lower-cased text)`` list from `build_turns`.
        verdict_at: flat utterance index -> 1/0 (the tutor-verdict proxy projected onto
            the utterance stream), so recovery is measured on a verdict, not on the absence
            of a marker.

    Returns:
        Dict of the 7 ``rep_*`` features (shares NaN and rep_has_arc 0 when there is no arc).
    """
    n = len(utts)
    kinds = {"prompt": 0, "explain": 0, "answer": 0}
    n_arcs = recovered = re_error = 0
    n_tutor = sum(1 for r, _ in utts if r == "teacher")
    for i, (role, low) in enumerate(utts):
        if role != "teacher" or not (CORRECTIVE.search(low) or _FEEDBACK_INCORRECT.search(low)):
            continue
        n_arcs += 1
        # strategy: the next tutor utterance in the arc (or the rest of this one)
        kind = _repair_kind(low, low)
        if kind is None:
            for j in range(i + 1, min(n, i + REPAIR_WINDOW)):
                if utts[j][0] == "teacher":
                    kind = _repair_kind(utts[j][1], utts[j][1])
                    break
        if kind:
            kinds[kind] += 1
        # recovery: the next graded student answer after this corrective
        for j in range(i + 1, n):
            v = verdict_at.get(j)
            if v is not None:
                recovered += v
                break
        # re-error: another corrective inside the window
        for j in range(i + 1, min(n, i + 1 + REPAIR_WINDOW)):
            if utts[j][0] == "teacher" and (CORRECTIVE.search(utts[j][1])
                                            or _FEEDBACK_INCORRECT.search(utts[j][1])):
                re_error += 1
                break
    if n_arcs == 0:
        return {"rep_prompt_share": np.nan, "rep_explain_share": np.nan,
                "rep_answer_share": np.nan, "rep_recovery_rate": np.nan,
                "rep_re_error_rate": np.nan, "rep_arcs_per_tutor_turn": 0.0,
                "rep_has_arc": 0.0}
    tot_kind = max(1, sum(kinds.values()))
    return {
        "rep_prompt_share": kinds["prompt"] / tot_kind,
        "rep_explain_share": kinds["explain"] / tot_kind,
        "rep_answer_share": kinds["answer"] / tot_kind,
        "rep_recovery_rate": recovered / n_arcs,
        "rep_re_error_rate": re_error / n_arcs,
        "rep_arcs_per_tutor_turn": n_arcs / max(1, n_tutor),
        "rep_has_arc": 1.0,
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    """Build the evidence blocks and the per-turn text table for every training transcript.

    Returns:
        None. Writes artifacts/evidence_features{suffix}.csv, artifacts/turn_text{suffix}.pkl.gz
        and figures/data/evidence_proxy_validation{suffix}.md.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap sessions (smoke test)")
    ap.add_argument("--suffix", default="", help="suffix for the output filenames")
    ap.add_argument("--trunc", type=int, nargs=3, default=(220, 220, 160),
                    metavar=("TEACHER", "STUDENT", "REPLY"),
                    help="per-turn text truncation stored in turn_text. The default is the "
                         "shipped value: every distilled classifier of the shipped model was "
                         "fitted under it, and its vectoriser vocabulary and idf are computed "
                         "over the truncated strings, so changing it invalidates the classifier, "
                         "its block, the XGBoost model and the blend weight. Use a --suffix "
                         "and refit; never change it in place.")
    args = ap.parse_args()
    global _TT, _TS, _TR
    _TT, _TS, _TR = args.trunc
    if tuple(args.trunc) != (220, 220, 160):
        print(f"WARNING: truncation {tuple(args.trunc)} is not the shipped one; classifiers "
              f"fitted on this table are not interchangeable with the shipped ones", flush=True)

    lab = pd.read_csv(LABELS_122B)
    lab = lab.drop_duplicates(["session_id", "turn"])
    lab122 = {s: dict(zip(g.turn, g.label))
              for s, g in lab[lab.label.isin(["true", "false"])].groupby("session_id", sort=False)}
    print(f"122B labels: {len(lab):,} turns, {lab.session_id.nunique():,} sessions; "
          f"{sum(len(v) for v in lab122.values()):,} answerable", flush=True)

    paths = sorted(TX.glob("*.csv"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"reading {len(paths):,} transcripts...", flush=True)

    rows = []
    turn_rows = []                     # per-turn text table for the distilled proxy
    agree = np.zeros((2, 2), int)      # [proxy][122b]
    prox_n = both_n = 0
    for pi, p in enumerate(paths):
        if pi % 4000 == 0:
            print(f"  {pi:,}/{len(paths):,}", flush=True)
        sid = p.stem
        tdf = pd.read_csv(p)
        turns, utts = build_turns(tdf)
        if not turns:
            rows.append({"session_id": sid})
            continue
        n_turns = len(turns)

        # --- per-turn text, for distilling the 122B annotator into a CPU model ---
        # Bounds come from --trunc; the default reproduces the shipped table exactly. The
        # truncation is short because the annotator's decision is local (does this reply
        # answer that question) and the whole table has to fit in memory downstream.
        for i, t in enumerate(turns):
            if not t["student"].strip():
                continue
            turn_rows.append((sid, t["turn"], i, n_turns,
                              t["teacher"][-_TT:], t["student"][:_TS],
                              (turns[i + 1]["teacher"] if i + 1 < len(turns) else "")[:_TR]))

        # --- proxy answer log + its utterance-index projection ------------------
        pseq, ppos = proxy_sequence(turns)
        prox_n += len(pseq)
        verdict_at = {}
        ui = 0                                   # walk utterances alongside turns
        turn_last_student_utt = {}
        cur_role = utts[0][0] if utts else None
        t_idx = turns[0]["turn"]
        for j, (role, _) in enumerate(utts):
            if role == "teacher" and cur_role == "student":
                t_idx += 1
            cur_role = role
            if role == "student":
                turn_last_student_utt[t_idx] = j
        for t, v in zip(ppos, pseq):
            j = turn_last_student_utt.get(turns[t]["turn"])
            if j is not None:
                verdict_at[j] = v

        # --- 122B answer log ---------------------------------------------------
        lm = lab122.get(sid, {})
        seq122, pos122 = [], []
        for i, t in enumerate(turns):
            v = lm.get(t["turn"])
            if v in ("true", "false"):
                seq122.append(1 if v == "true" else 0)
                pos122.append(i)

        # --- proxy vs 122B agreement (same turn index) -------------------------
        pmap = {turns[t]["turn"]: v for t, v in zip(ppos, pseq)}
        for tn, v in pmap.items():
            g = lm.get(tn)
            if g in ("true", "false"):
                agree[v][1 if g == "true" else 0] += 1
                both_n += 1

        r = {"session_id": sid}
        r |= record_block(seq122, pos122, n_turns, "rec122")
        r |= record_block(pseq, ppos, n_turns, "recprx")
        r |= contingency_block(utts)
        r |= repair_block(utts, verdict_at)
        rows.append(r)

    out = pd.DataFrame(rows)
    sfx = args.suffix
    dst = ROOT / f"artifacts/evidence_features{sfx}.csv"
    out.to_csv(dst, index=False)
    print(f"\nwrote {dst}  ({out.shape[0]:,} sessions x {out.shape[1] - 1} features)")

    tt = pd.DataFrame(turn_rows, columns=["session_id", "turn", "pos", "n_turns",
                                          "tutor_q", "student", "tutor_reply"])
    tt.to_pickle(ROOT / f"artifacts/turn_text{sfx}.pkl.gz", compression="gzip")
    print(f"wrote artifacts/turn_text{sfx}.pkl.gz  ({len(tt):,} student turns)")

    # ---------------------------------------------------------- proxy validation
    tot = agree.sum()
    acc = (agree[0, 0] + agree[1, 1]) / max(1, tot)
    po = acc
    pe = ((agree[0].sum() * agree[:, 0].sum() + agree[1].sum() * agree[:, 1].sum())
          / max(1, tot ** 2))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    n122 = sum(len(v) for v in lab122.values())
    txt = [
        "# Tutor-verdict proxy vs the 122B per-turn correctness annotation\n\n",
        "The shippable half of block R has to read the answer log off regex, because a\n",
        "122B annotator cannot run inside the 6h offline container. The proxy grades\n",
        "student turn *t* from the tutor's reply in turn *t+1* (corrective => incorrect,\n",
        "confirmation => correct, neither => unlabelled).\n\n",
        f"- proxy-labelled turns: **{prox_n:,}**\n",
        f"- 122B-labelled turns: **{n122:,}**\n",
        f"- overlap (both labelled the same turn): **{both_n:,}** "
        f"({both_n / max(1, prox_n):.1%} of proxy, {both_n / max(1, n122):.1%} of 122B)\n",
        f"- **agreement on the overlap: {acc:.3f}**, Cohen's kappa **{kappa:.3f}**\n\n",
        "| | 122B false | 122B true |\n|---|---|---|\n",
        f"| proxy incorrect | {agree[0,0]:,} | {agree[0,1]:,} |\n",
        f"| proxy correct | {agree[1,0]:,} | {agree[1,1]:,} |\n",
    ]
    (ROOT / f"figures/data/evidence_proxy_validation{sfx}.md").write_text("".join(txt))
    print("".join(txt))


if __name__ == "__main__":
    main()
