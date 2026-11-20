"""Aggregations of a per-turn soft-label sequence into session- and objective-level features.

One code path, two callers. `scripts/build_v2_families.py` calls these functions on the
per-turn probability tables written by `scripts/distil_v2_moves.py`; `magnificat.v2_features`
calls the same functions on probabilities predicted from an unseen transcript. The
aggregation is a pure function of ``(T, S, pos, n)``, so there is nothing to re-implement on
the inference side, and nothing that can drift.

Two constraints shape the definitions below:

1. No cross-sample statistics. Test samples must be processed independently, so no quantity
   may be computed across the batch being scored. Every constant here is fixed in the code; no
   quantile, mean or scale is read from the data. Each function takes one session's arrays and
   returns one dict.
2. Register transfer. The training corpus is ~100-turn transcribed voice tutoring, and the
   features should also hold on short typed chat. So features are rates and ratios, never
   counts; every denominator is either a probability mass
   with an additive guard or a turn count; and every unbounded ratio is clipped at a fixed cap,
   for the reason `distil_v2_moves.ELICIT_TELL_CAP` documents: an XGBoost model is invariant to
   monotone transforms, so `log1p` would change nothing, but split thresholds are learned in
   TtA's units and a shorter register runs past them.

The within-session problem. A session-level scalar is constant across the objectives that
share a transcript, while the task is to tell those objectives apart. :func:`lo_window_features`
(with :func:`lo_soft_coverage`) is the only family here that varies within a session; the
others can only shift the session's base level.
"""
from __future__ import annotations

import numpy as np

# ── code inventories ────────────────────────────────────────────────────────────────────
#: Tutor residuals per the decision order of src/training/docs/39_moves_annotations_v2.md §5,
#: steps 2-4 and 6. `TUTOR_OTHER` is not here: its definition is "the turn IS pedagogically
#: meaningful, but no code above fits", so it belongs on the instructional side of the
#: denominator.
TUTOR_RESID = ["TUTOR_SOCIAL", "TUTOR_TECHNICAL", "TUTOR_UNINTELLIGIBLE", "TUTOR_NO_MOVE"]
STUDENT_RESID = ["STUDENT_SOCIAL", "STUDENT_TECHNICAL", "STUDENT_UNINTELLIGIBLE",
                 "STUDENT_NO_MOVE"]

#: Elicit / tell, as `distil_v2_moves.RATIO_SPEC["v2"]` defines them; copied by value rather
#: than imported because `scripts/` is not importable inside the runtime container.
ELICIT = ["PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP", "PROMPTING_RELATED_CONCEPTS",
          "PROMPTING_ALTERNATIVE_REPRESENTATION", "PROMPTING_SELF_CORRECTION",
          "RELEASE_HANDOFF"]
TELL = ["EXPLAINING_CONCEPTUAL", "EXPLAINING_PROCEDURAL", "GIVING_ANSWER",
        "WORKED_DEMONSTRATION"]
#: Interrogative moves of any kind: the "is the tutor asking?" axis, the union of the
#: interrogative codes rather than `ASKING_QUESTION` alone.
ASK = ELICIT + ["ASKING_QUESTION", "TRANSFER_PROBE", "OFFERING_CHOICE"]
#: Student engagement / struggle axes.
STU_ENGAGE = ["REASONING_SHOWN", "SELF_CORRECTING", "INSIGHT_DELIGHT"]
STU_STRUGGLE = ["CONFUSION_EXPRESSED", "FRUSTRATION", "SEEKING_CLARIFICATION"]

EPS = 1e-3
#: Same value and same reason as `distil_v2_moves.ELICIT_TELL_CAP` / `v2_features`.
RATIO_CAP = 10.0


def _idx(cols: list[str]) -> dict:
    return {c: j for j, c in enumerate(cols)}


def _mass(P: np.ndarray, cols: list[str], names: list[str]) -> np.ndarray:
    """Sum each turn's probability over a named subset of codes.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column of ``P``.
        names: codes to sum; names not in ``cols`` are ignored.

    Returns:
        Array of shape (n,); zeros if no name is present.
    """
    ix = _idx(cols)
    js = [ix[c] for c in names if c in ix]
    return P[:, js].sum(axis=1) if js else np.zeros(P.shape[0])


def instructional_weight(P: np.ndarray, cols: list[str], resid: list[str]) -> np.ndarray:
    """Compute the per-turn instructional mass ``w_t = clip(1 - sum(residual codes), 0, 1)``.

    The heads are independent one-vs-rest logistic regressions, so a row does not sum to 1 and
    the residual mass can exceed 1; clipping makes ``w`` a weight rather than a signed
    quantity. The residual codes are those of src/training/docs/39_moves_annotations_v2.md §5.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column of ``P``.
        resid: the residual (non-instructional) codes of this role.

    Returns:
        Array of shape (n,) with values in [0, 1].
    """
    return np.clip(1.0 - _mass(P, cols, resid), 0.0, 1.0)


def _ratio(num: float, den: float) -> float:
    """Return ``min(num / (den + EPS), RATIO_CAP)`` as a float.

    The cap is a constant: deriving it from the batch would make the feature depend on the
    other samples being scored, which the competition rules forbid.
    """
    return float(min(num / (den + EPS), RATIO_CAP))


# ── family: plain / instructional-denominator / trimmed rates ───────────────────────────
def rates(P: np.ndarray, cols: list[str], weights: np.ndarray | None = None,
          mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the (weighted) mean of the soft probabilities per code.

    This is the expectation of the oracle's ``hits / n_turns``, which makes the block
    length-normalised. ``weights=None`` and ``mask=None`` reproduce the
    ``P.mean(axis=0)`` of :func:`magnificat.v2_features.build_v2_matrix` exactly.

    Args:
        P: per-turn probabilities, shape (n, n_codes).
        cols: code name of each column (unused; kept for a uniform family signature).
        weights: optional per-turn weights, shape (n,), e.g. :func:`instructional_weight`.
            If they sum to at most ``EPS`` the unweighted mean is returned.
        mask: optional boolean mask selecting the turns to average.

    Returns:
        Array of shape (n_codes,); all NaN when no turn is selected.
    """
    if mask is not None:
        if not mask.any():
            return np.full(P.shape[1], np.nan)
        P = P[mask]
        weights = None if weights is None else weights[mask]
    if P.shape[0] == 0:
        return np.full(P.shape[1], np.nan)
    if weights is None:
        return P.mean(axis=0)
    den = float(weights.sum())
    if den <= EPS:                       # a session with no instructional mass at all
        return P.mean(axis=0)            # fall back rather than emit a 0/0 sentinel
    return (P * weights[:, None]).sum(axis=0) / den


def instructional_span(w: np.ndarray, thresh: float = 0.5) -> tuple[int, int]:
    """Find the first and last turn whose instructional mass exceeds ``thresh``.

    The labels set the boundary: the codebook says directly which turns were teaching. Falls
    back to the whole session when fewer than two turns qualify, so an 8-turn typed exchange is
    never trimmed to nothing.

    Args:
        w: per-turn instructional mass from :func:`instructional_weight`.
        thresh: minimum mass for a turn to count as teaching.

    Returns:
        Tuple ``(first, last)`` of turn indices, inclusive.
    """
    hit = np.where(w > thresh)[0]
    if hit.size < 2:
        return 0, len(w) - 1
    return int(hit[0]), int(hit[-1])




# ── generalisation: make a rate's noise independent of transcript length ────────────────
#
# This is the `shrunk` family (`shrunk_rates` with the global `SHRINK_KAPPA`), the aggregation
# of the taxonomy-v2 block the shipped XGBoost model uses (`v2ht_` / `v2hs_`).
#
# A rate is `mean_t P[t,c]`, so its sampling sd is `sigma/sqrt(n)`. TtA sessions have a median
# of 102 turns and only 0.5% below 17, so every split threshold the booster learns is fitted
# where n~102. `shrunk_rates` shrinks each rate toward a per-code prior with weight `kappa`,
#
#     r'_c = (n * r_c + kappa * pi_c) / (n + kappa)      sd(r'_c) = sigma * sqrt(n) / (n + kappa)
#
# and that sd is nearly flat in n rather than falling as 1/sqrt(n), with no added columns.
#
# Two properties:
#   * `pi_c` is a fixed per-code constant, frozen below at build time. It is not a per-LO
#     statistic (which would not transfer to unseen objectives) and is never computed from the
#     batch being scored.
#   * TtA's shrink factor n/(n+kappa) spans 0.66 (p5) to 0.87 (p95), so within TtA this is
#     nearly an affine per-code rescaling, to which an XGBoost model is invariant. An 8-turn
#     session shrinks by 0.24. The effect is therefore concentrated on short transcripts.
#
#: Prior "sample size" in turns: how much evidence a session needs before its own rates outweigh
#: the corpus prior. A fixed constant: the value that flattens sd(r') over 8..162 turns, where
#: sqrt(n)/(n+kappa) peaks at n=kappa.
SHRINK_KAPPA = 25.0

#: Per-code corpus means over all 2.36M/2.37M scored turns, frozen at build time.
#: Regenerate only when the distilled classifiers change, and record it when you do.
CODE_PRIOR = {
    "ANSWER_ONLY": 0.268468,
    "ASKING_QUESTION": 0.303358,
    "CONFUSION_EXPRESSED": 0.037346,
    "EXPLAINING_CONCEPTUAL": 0.036067,
    "EXPLAINING_PROCEDURAL": 0.058784,
    "EXPLAINING_TOOL": 0.009637,
    "FEEDBACK_CORRECT": 0.088575,
    "FEEDBACK_INCORRECT": 0.018471,
    "FEEDBACK_NEUTRAL": 0.028676,
    "FRUSTRATION": 0.003203,
    "GIVING_ANSWER": 0.014881,
    "GIVING_EXAMPLE": 0.006623,
    "GIVING_HINT": 0.034988,
    "GIVING_PRAISE": 0.076777,
    "GUIDING_SESSION": 0.06555,
    "INSIGHT_DELIGHT": 0.012104,
    "MOTIVATING_RELEVANCE": 0.002032,
    "NORMALIZING_DIFFICULTY": 0.010319,
    "OFFERING_CHOICE": 0.005881,
    "PROMPTING_ALTERNATIVE_REPRESENTATION": 0.005898,
    "PROMPTING_NEXT_STEP": 0.020814,
    "PROMPTING_RELATED_CONCEPTS": 0.010894,
    "PROMPTING_SELF_CORRECTION": 0.015552,
    "PROMPTING_SELF_EXPLANATION": 0.078456,
    "REASONING_SHOWN": 0.169928,
    "RELEASE_HANDOFF": 0.053235,
    "RESTATING": 0.011958,
    "REVOICING": 0.017075,
    "SEEKING_CLARIFICATION": 0.06531,
    "SELF_CORRECTING": 0.018238,
    "SIMPLIFYING_TO_SUBPROBLEM": 0.001225,
    "STUDENT_NO_MOVE": 0.183111,
    "STUDENT_OTHER": 0.036314,
    "STUDENT_SOCIAL": 0.074467,
    "STUDENT_TECHNICAL": 0.04102,
    "STUDENT_UNINTELLIGIBLE": 0.089521,
    "SUMMARIZING_PROGRESS": 0.01146,
    "TRANSFER_PROBE": 0.001369,
    "TUTOR_NO_MOVE": 0.009028,
    "TUTOR_OTHER": 0.019509,
    "TUTOR_SOCIAL": 0.058053,
    "TUTOR_TECHNICAL": 0.048656,
    "TUTOR_UNINTELLIGIBLE": 0.018144,
    "WORKED_DEMONSTRATION": 0.015801,
}




#: Per-code empirical-Bayes shrinkage, by method of moments on training statistics only.
#: `r_s` is a mean of `n_s` per-turn probabilities, so
#:     Var(r_s) = Var_between + E[sigma2_within / n_s]
#: and the variance-optimal weight on a session's own estimate is
#:     w_s = Var_between / (Var_between + sigma2_within/n_s) = n_s / (n_s + k_c),
#:     k_c = sigma2_within,c / Var_between,c.
#: It is frozen at build time like CODE_PRIOR. Used by the `shrunkeb` family only; the
#: shipped `shrunk` family uses the global SHRINK_KAPPA.
CODE_KAPPA = {
    "ANSWER_ONLY": 14.838,
    "ASKING_QUESTION": 14.621,
    "CONFUSION_EXPRESSED": 51.87,
    "EXPLAINING_CONCEPTUAL": 25.134,
    "EXPLAINING_PROCEDURAL": 23.857,
    "EXPLAINING_TOOL": 32.758,
    "FEEDBACK_CORRECT": 21.494,
    "FEEDBACK_INCORRECT": 49.819,
    "FEEDBACK_NEUTRAL": 28.583,
    "FRUSTRATION": 12.337,
    "GIVING_ANSWER": 27.285,
    "GIVING_EXAMPLE": 55.568,
    "GIVING_HINT": 32.585,
    "GIVING_PRAISE": 23.369,
    "GUIDING_SESSION": 50.924,
    "INSIGHT_DELIGHT": 67.808,
    "MOTIVATING_RELEVANCE": 60.884,
    "NORMALIZING_DIFFICULTY": 58.953,
    "OFFERING_CHOICE": 91.732,
    "PROMPTING_ALTERNATIVE_REPRESENTATION": 62.618,
    "PROMPTING_NEXT_STEP": 32.08,
    "PROMPTING_RELATED_CONCEPTS": 49.314,
    "PROMPTING_SELF_CORRECTION": 63.444,
    "PROMPTING_SELF_EXPLANATION": 32.473,
    "REASONING_SHOWN": 15.266,
    "RELEASE_HANDOFF": 44.6,
    "RESTATING": 36.878,
    "REVOICING": 43.853,
    "SEEKING_CLARIFICATION": 41.465,
    "SELF_CORRECTING": 50.411,
    "SIMPLIFYING_TO_SUBPROBLEM": 13.34,
    "STUDENT_NO_MOVE": 16.987,
    "STUDENT_OTHER": 21.958,
    "STUDENT_SOCIAL": 72.645,
    "STUDENT_TECHNICAL": 10.922,
    "STUDENT_UNINTELLIGIBLE": 16.786,
    "SUMMARIZING_PROGRESS": 63.047,
    "TRANSFER_PROBE": 16.252,
    "TUTOR_NO_MOVE": 63.382,
    "TUTOR_OTHER": 23.823,
    "TUTOR_SOCIAL": 91.321,
    "TUTOR_TECHNICAL": 10.131,
    "TUTOR_UNINTELLIGIBLE": 9.074,
    "WORKED_DEMONSTRATION": 39.323,
}


def eb_shrunk_rates(P, cols, mask=None):
    """Compute rates shrunk toward :data:`CODE_PRIOR` with a per-code kappa (:data:`CODE_KAPPA`).

    Same shape as :func:`shrunk_rates`, different weights.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column; a code missing from the tables uses the session's own
            mean as prior and :data:`SHRINK_KAPPA` as kappa.
        mask: optional boolean mask selecting the turns to use.

    Returns:
        Array of shape (len(cols),); all NaN when no turn is selected.
    """
    if mask is not None:
        P = P[mask]
    n = P.shape[0]
    if n == 0:
        return np.full(len(cols), np.nan)
    tot = P.sum(axis=0)
    pri = np.array([CODE_PRIOR.get(c, float(tot[j]/max(n,1))) for j, c in enumerate(cols)])
    kap = np.array([CODE_KAPPA.get(c, SHRINK_KAPPA) for c in cols])
    return (tot + kap*pri) / (n + kap)


def shrunk_rates(P, cols, mask=None, kappa: float = SHRINK_KAPPA):
    """Compute rates shrunk toward the frozen per-code prior: same columns, length-stable noise.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column; a code missing from :data:`CODE_PRIOR` uses the
            session's own mean as prior.
        mask: optional boolean mask selecting the turns to use.
        kappa: prior weight in turns.

    Returns:
        Array ``(sum_t P + kappa * prior) / (n + kappa)`` of shape (len(cols),); all NaN when
        no turn is selected.
    """
    if mask is not None:
        P = P[mask]
    n = P.shape[0]
    if n == 0:
        return np.full(len(cols), np.nan)
    tot = P.sum(axis=0)
    pri = np.array([CODE_PRIOR.get(c, float(tot[j] / max(n, 1)))
                    for j, c in enumerate(cols)])
    return (tot + kappa * pri) / (n + kappa)




# ── window rates: make the denominator constant ─────────────────────────────────────────
#
# Aggregate over the last K turns rather than over all n, so for every session with at least K
# turns the estimator averages exactly K terms and its sampling sd does not depend on length.
# Sessions shorter than K use all their turns.
#
#: Default window: the annotation chunk size used throughout this pipeline
#: (`annotate_v2_moves` / `distil_v2_moves`). `v2_features` also offers 12 and 48 (`win12`,
#: `win48`).
WINDOW_K = 24


def window_rates(P: np.ndarray, cols: list[str], mask: np.ndarray | None = None,
                 k: int = WINDOW_K) -> np.ndarray:
    """Average the probabilities over the last ``k`` turns of this role (all, if fewer exist).

    Length-invariant by construction for any session with >= k turns of the role.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column.
        mask: optional boolean mask selecting the turns of this role.
        k: window length in turns.

    Returns:
        Array of shape (len(cols),); all NaN when no turn is selected.
    """
    if mask is not None:
        P = P[mask]
    if P.shape[0] == 0:
        return np.full(len(cols), np.nan)
    return P[-k:].mean(axis=0)


# ── family: boundary / shape of the session ─────────────────────────────────────────────
def boundary_features(wT: np.ndarray, wS: np.ndarray, T: np.ndarray, tcols: list[str],
                      pre: str) -> dict:
    """Describe where the teaching starts and stops, and how densely it fills the session.

    All six columns are fractions of session length, so they survive the change of register.
    ``turns_to_ask`` is how long before the tutor first asks anything, also as a fraction of
    the session rather than a turn count.

    Args:
        wT: per-turn tutor instructional mass (:func:`instructional_weight`).
        wS: per-turn student instructional mass (unused here; the caller adds its mean).
        T: tutor probabilities, shape (n, len(tcols)).
        tcols: tutor code name of each column of ``T``.
        pre: column prefix.

    Returns:
        Dict with ``{pre}_warmup_frac``, ``_closing_frac``, ``_instr_span_frac``,
        ``_instr_share``, ``_turns_to_ask`` and ``_code_mass``; all NaN for an empty session.
    """
    n = len(wT)
    if n == 0:
        return {f"{pre}_{k}": np.nan for k in
                ("warmup_frac", "closing_frac", "instr_span_frac", "instr_share",
                 "turns_to_ask", "code_mass")}
    lo, hi = instructional_span(wT)
    ask = _mass(T, tcols, ASK)
    first_ask = np.where(ask > 0.5)[0]
    return {
        f"{pre}_warmup_frac": lo / n,
        f"{pre}_closing_frac": (n - 1 - hi) / n,
        f"{pre}_instr_span_frac": (hi - lo + 1) / n,
        f"{pre}_instr_share": float(wT.mean()),
        # 1.0 = the tutor never asks anything, which is a real and different state from
        # "asks on the last turn"; a NaN here would be read as missing data instead.
        f"{pre}_turns_to_ask": (first_ask[0] / n) if first_ask.size else 1.0,
        # Mean total code mass per turn. >1 means the annotator saw multi-function turns;
        # this is the soft analogue of the `code_2` second label, without a second label slot.
        f"{pre}_code_mass": float(T.sum(axis=1).mean()),
    }


# ── family: direction of travel ─────────────────────────────────────────────────────────
def slope_features(P: np.ndarray, cols: list[str], axes: dict, pre: str) -> dict:
    """Compute ``late - early`` (last third minus first third) for a few axes.

    Computed on aggregate axes, not per code. An axis is a named subset summed per
    turn, so the quantity is still a rate in [0, k].

    Thirds are by turn index, not by time: `timestamp` is an elapsed offset and turn rhythm
    varies, but a third of the turns is a third of the turns in any register.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column.
        axes: ``{axis_name: [codes]}``.
        pre: column prefix.

    Returns:
        Dict ``{pre}_{axis}_slope`` per axis; NaN when the session has fewer than 3 turns.
    """
    n = P.shape[0]
    out = {}
    if n < 3:
        return {f"{pre}_{k}_slope": np.nan for k in axes}
    c = n // 3
    for name, names in axes.items():
        m = _mass(P, cols, names)
        out[f"{pre}_{name}_slope"] = float(m[-c:].mean() - m[:c].mean())
    return out


# ── family: dispersion in time ──────────────────────────────────────────────────────────
def dispersion_features(P: np.ndarray, cols: list[str], axes: dict, pre: str) -> dict:
    """Measure whether each axis is spread through the session or bunched, and where it sits.

    ``spread`` is the entropy of the axis's temporal distribution ``q_t = p_t / sum(p)``,
    normalised by ``log n`` so it lies in [0, 1] for any session length (1 is perfectly even,
    0 is a single burst). ``centroid`` is the mass-weighted mean position in [0, 1]. Both are
    invariant to how much of the axis there is, which separates them from the rate: the same
    questioning rate delivered in one burst or steadily throughout is a different lesson.

    Args:
        P: per-turn probabilities, shape (n, len(cols)).
        cols: code name of each column.
        axes: ``{axis_name: [codes]}``.
        pre: column prefix.

    Returns:
        Dict with ``{pre}_{axis}_spread`` and ``{pre}_{axis}_centroid`` per axis; NaN when
        the session has fewer than 2 turns or the axis has no mass.
    """
    n = P.shape[0]
    out = {}
    for name, names in axes.items():
        m = _mass(P, cols, names)
        tot = float(m.sum())
        if n < 2 or tot <= EPS:
            out[f"{pre}_{name}_spread"] = np.nan
            out[f"{pre}_{name}_centroid"] = np.nan
            continue
        q = m / tot
        h = -np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-12)), 0.0))
        out[f"{pre}_{name}_spread"] = float(h / np.log(n))
        out[f"{pre}_{name}_centroid"] = float((q * (np.arange(n) / (n - 1))).sum())
    return out


# ── family: transitions and contingency ─────────────────────────────────────────────────
#
# The paired-turn layout makes these cheap and unambiguous. `turn_text.pkl.gz` stores each turn
# as (tutor_q, student, tutor_reply) with `tutor_reply[t] == tutor_q[t+1]`, so within a row the
# tutor move precedes the student move, and the tutor's response to the student at row t is the
# tutor move at row t+1. Forward pairs are read from one row, backward pairs from consecutive
# rows, with no window search or alignment assumption.
#
# Each of these is a conditional rate: the denominator is how often the antecedent fired, so
# "praise given the student just reasoned" is a different number from "praise", and that
# difference is what the construct measures.

#: ``name: (antecedent codes, consequent codes, direction)``. ``"t2s"`` reads the student slot
#: of the same row; ``"s2t"`` reads the tutor slot of the next row.
TRANSITIONS = {
    # elicitation that lands vs elicitation that misfires: the elicit-vs-tell axis, measured
    # by what the student does next rather than by how often the tutor elicits
    "elicit_to_reason":  (ELICIT, ["REASONING_SHOWN"], "t2s"),
    "elicit_to_confuse": (ELICIT, STU_STRUGGLE, "t2s"),
    "elicit_to_bare":    (ELICIT, ["ANSWER_ONLY"], "t2s"),
    # what the tutor does when the student is stuck: scaffold or take over
    "confuse_to_tell":   (STU_STRUGGLE, TELL, "s2t"),
    "confuse_to_scaffold": (STU_STRUGGLE,
                            ["GIVING_HINT", "PROMPTING_SELF_CORRECTION",
                             "SIMPLIFYING_TO_SUBPROBLEM", "NORMALIZING_DIFFICULTY"], "s2t"),
    # transfer of the work, and whether it is taken up
    "handoff_to_reason": (["RELEASE_HANDOFF"], STU_ENGAGE, "t2s"),
    # repair: whether naming an error produces a self-correction
    "wrong_to_repair":   (["FEEDBACK_INCORRECT"], ["SELF_CORRECTING", "REASONING_SHOWN"],
                          "t2s"),
    # contingent praise: praise after the student reasoned, not praise in general
    "reason_to_praise":  (STU_ENGAGE, ["GIVING_PRAISE", "FEEDBACK_CORRECT"], "s2t"),
}


def transition_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                        pre: str) -> dict:
    """Compute the soft conditional rates of :data:`TRANSITIONS`.

    Each rate is ``sum_t a_t * c_t / (sum_t a_t + eps)``, bounded in [0, k]. Soft rather than
    thresholded: a 32-way annotator expresses most of what it knows below any threshold.

    Args:
        T: tutor probabilities on the paired-turn axis, shape (n, len(tcols)).
        tcols: tutor code name of each column of ``T``.
        S: student probabilities on the same axis, shape (n, len(scols)).
        scols: student code name of each column of ``S``.
        pre: column prefix.

    Returns:
        Dict ``{pre}_{name}`` per transition; NaN when there are fewer than 2 turns or the
        antecedent never fires.
    """
    n = T.shape[0]
    out = {}
    for name, (ante, cons, direction) in TRANSITIONS.items():
        if n < 2:
            out[f"{pre}_{name}"] = np.nan
            continue
        if direction == "t2s":                      # tutor row t -> student row t
            a, c = _mass(T, tcols, ante), _mass(S, scols, cons)
        else:                                       # "s2t": student row t -> tutor row t+1
            a, c = _mass(S, scols, ante)[:-1], _mass(T, tcols, cons)[1:]
        den = float(a.sum())
        out[f"{pre}_{name}"] = _ratio(float((a * c).sum()), den) if den > EPS else np.nan
    return out


# ── family: bounded ratios ──────────────────────────────────────────────────────────────
def ratio_features(P: np.ndarray, cols: list[str], pre: str) -> dict:
    """Compute elicit-vs-tell and scaffold-vs-give-away as shares ``a / (a + b)``, not ``a / b``.

    In-domain the two forms are equivalent: an XGBoost model is invariant to monotone
    transforms of a single feature, and ``x -> x/(1+x)`` is monotone, so the share and the
    quotient induce identical splits on TtA. Off-corpus they are not the same feature. The
    quotient is unbounded and its tail is set by an additive guard. A corpus where tutors
    explain less shrinks the denominator further and the value runs past every threshold
    learned here, so a capped quotient (`distil_v2_moves.ELICIT_TELL_CAP`) collapses a whole
    region of the input onto one leaf. The share cannot leave [0, 1], so every threshold
    learned on TtA still partitions the range on a typed corpus.

    Args:
        P: tutor probabilities, shape (n, len(cols)).
        cols: tutor code name of each column.
        pre: column prefix.

    Returns:
        Dict with ``{pre}_elicit_share``, ``{pre}_ask_share`` and ``{pre}_scaffold_share``.
    """
    el, te = _mass(P, cols, ELICIT).sum(), _mass(P, cols, TELL).sum()
    ask = _mass(P, cols, ASK).sum()
    scaf = _mass(P, cols, ["GIVING_HINT", "SIMPLIFYING_TO_SUBPROBLEM",
                           "PROMPTING_SELF_CORRECTION", "NORMALIZING_DIFFICULTY"]).sum()
    give = _mass(P, cols, ["GIVING_ANSWER", "WORKED_DEMONSTRATION"]).sum()
    return {f"{pre}_elicit_share": float(el / (el + te + EPS)),
            f"{pre}_ask_share": float(ask / (ask + te + EPS)),
            f"{pre}_scaffold_share": float(scaf / (scaf + give + EPS))}


# ── family: how decisively the annotator could code this transcript ─────────────────────
def quality_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                     tmask: np.ndarray, pre: str) -> dict:
    """Compute register and legibility proxies, all in [0, 1] and length-normalised.

    These measure how codable the transcript is, not the tutoring. `decisive` is the mean top-1
    probability per turn and `entropy` the mean normalised entropy across codes: together they
    say whether the annotator saw clear moves or a smear. Transcribed voice and typed chat
    differ on exactly this axis (0 of the 22,821 training sessions are short and clean), and
    none of the rate columns can express it: a session that is 30% `TUTOR_UNINTELLIGIBLE` and
    one the annotator could not read produce different smears at the same rate.

    ``tutor_turn_share`` is the fraction of paired turns carrying a tutor utterance, a
    diarisation and turn-taking property rather than a pedagogical one.

    Args:
        T: tutor probabilities on the paired-turn axis, shape (n, len(tcols)).
        tcols: tutor code name of each column of ``T``.
        S: student probabilities on the same axis, shape (n, len(scols)).
        scols: student code name of each column of ``S``.
        tmask: boolean mask of turns that carry a tutor utterance.
        pre: column prefix.

    Returns:
        Dict with ``{pre}_tutor_decisive``, ``_tutor_entropy``, ``_student_decisive``,
        ``_student_entropy`` and ``_tutor_turn_share`` (NaN where undefined).
    """
    out = {}
    for name, (P, mask) in (("tutor", (T, tmask)), ("student", (S, None))):
        Q = P[mask] if mask is not None else P
        if Q.shape[0] == 0 or Q.shape[1] < 2:
            out[f"{pre}_{name}_decisive"] = np.nan
            out[f"{pre}_{name}_entropy"] = np.nan
            continue
        out[f"{pre}_{name}_decisive"] = float(Q.max(axis=1).mean())
        tot = Q.sum(axis=1, keepdims=True)
        q = Q / np.maximum(tot, 1e-9)
        h = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1) / np.log(Q.shape[1])
        # a turn with no mass at all is uninformative, not maximally uncertain
        out[f"{pre}_{name}_entropy"] = float(h[tot[:, 0] > 1e-6].mean()) \
            if (tot[:, 0] > 1e-6).any() else np.nan
    out[f"{pre}_tutor_turn_share"] = float(tmask.mean()) if tmask.size else np.nan
    return out


# ── family: soft LO proximity (one column, and one of two that vary by objective) ─────────
#: Decay constant for the proximity kernel, in turns; a fixed constant. tau=1 makes the kernel
#: effectively "this turn or the two either side", which the hard +/-2 window approximates
#: with a step function.
LO_TAU = 1.0
#: Below this many turns the mean of a decay kernel is dominated by the session being short
#: rather than by where the objective sits, so the column reports NaN rather than a number the
#: booster would read as "the objective is everywhere".
LO_MIN_TURNS = 6


def lo_soft_coverage(hits: np.ndarray, n: int, pre: str, tau: float = LO_TAU) -> dict:
    """Compute the soft coverage of an objective, ``c_LO = (1/n) * sum_t exp(-d_t / tau)``.

    ``d_t`` is the distance in turns from turn ``t`` to the nearest turn whose words hit the
    objective's keywords.

    One column by design.

    Together with :func:`lo_window_features` it is the only quantity in this module that is a
    function of the objective as well as the transcript, so it can separate two responses that
    share a session.

    NaN, never zero, when the objective is never named or the session is too short: "never
    discussed" and "discussed nowhere near here" are different states, and a zero would be read
    as the second.

    Args:
        hits: boolean mask over the family turn axis, True where a turn's words hit the
            objective's keywords; None if there is no hit.
        n: number of turns on the axis.
        pre: column prefix.
        tau: decay constant in turns.

    Returns:
        Dict ``{pre}_soft_cov``: float in (0, 1], or NaN.
    """
    if n < LO_MIN_TURNS or hits is None or not hits.any():
        return {f"{pre}_soft_cov": np.nan}
    idx = np.arange(n)
    hit_at = idx[hits]
    d = np.abs(idx[:, None] - hit_at[None, :]).min(axis=1)
    return {f"{pre}_soft_cov": float(np.exp(-d / tau).mean())}


# ── family: the LO-relevant window (the only one that varies within a session) ───────────
def lo_window_features(T: np.ndarray, tcols: list[str], S: np.ndarray, scols: list[str],
                       win: np.ndarray | None, pre: str) -> dict:
    """Compute rates on the turns that handled this objective, and how they differ from the rest.

    ``win`` is a boolean mask over turns, supplied by the caller so that the keyword rule lives
    in one place (`histpos_features._lo_words`, the rule `features._lo_transcript_stats` uses)
    rather than being re-typed here.

    The rates are reported as inside-minus-outside gaps, not as raw inside rates. A raw inside
    rate is mostly the session's level, which the other columns already carry; the gap is the
    part specific to this objective, and it is what can separate two responses that share a
    transcript. When the objective is never mentioned, all columns are
    NaN; when the window covers the whole session, the gap columns are NaN. NaN is the state
    the booster already meets for a transcript with no turns; a zero would read as
    "no difference".

    Args:
        T: tutor probabilities on the family turn axis, shape (n, len(tcols)).
        tcols: tutor code name of each column of ``T``.
        S: student probabilities on the same axis, shape (n, len(scols)).
        scols: student code name of each column of ``S``.
        win: boolean window mask over the axis, or None.
        pre: column prefix.

    Returns:
        Dict with ``{pre}_cover``, ``_pos``, ``_gap_{ask,tell,reason,struggle,instr}`` and
        ``_end_{reason,struggle}``.
    """
    keys = ("cover", "pos", "gap_ask", "gap_tell", "gap_reason", "gap_struggle", "gap_instr",
            "end_reason", "end_struggle")
    if win is None or T.shape[0] == 0 or not win.any():
        return {f"{pre}_{k}": np.nan for k in keys}
    n = T.shape[0]
    out = {f"{pre}_cover": float(win.mean()),
           # where in the session this objective was handled (the "distance" feature). An
           # objective taught in the last quarter is a different prediction problem from one
           # taught first and then left alone for 40 turns.
           f"{pre}_pos": float(np.arange(n)[win].mean() / max(1, n - 1))}
    axes = {"ask": (T, tcols, ASK), "tell": (T, tcols, TELL),
            "reason": (S, scols, STU_ENGAGE), "struggle": (S, scols, STU_STRUGGLE),
            "instr": (T, tcols, None)}
    out_mask = ~win
    for name, (P, cols, names) in axes.items():
        m = (instructional_weight(P, cols, TUTOR_RESID) if names is None
             else _mass(P, cols, names))
        if not out_mask.any():
            out[f"{pre}_gap_{name}"] = np.nan       # window is the whole session: no contrast
        else:
            out[f"{pre}_gap_{name}"] = float(m[win].mean() - m[out_mask].mean())

    # How the objective was left, not how it went on average: a student still struggling on
    # the last turn that touched this objective is a different prediction from one who
    # struggled in the middle and resolved it. Final quarter of the window, at least one turn,
    # so an 8-turn typed exchange still yields a value.
    wi = np.where(win)[0]
    tail = wi[-max(1, len(wi) // 4):]
    for name, (P, cols, names) in (("reason", (S, scols, STU_ENGAGE)),
                                   ("struggle", (S, scols, STU_STRUGGLE))):
        out[f"{pre}_end_{name}"] = float(_mass(P, cols, names)[tail].mean())
    return out
