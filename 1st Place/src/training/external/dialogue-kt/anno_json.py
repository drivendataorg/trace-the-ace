"""Sampling settings and tolerant JSON parsing for the vLLM annotators.

Decoding is constrained to the JSON schema unless SO_OFF=1 (plain greedy sampling with tolerant
parsing).

When the grammar's FSM cannot advance on a token, vLLM does not raise. It produces unparseable
text; a caller whose `except` falls back to a default then reports a plausible score with no
parse failures. The functions here therefore count every failure and report it instead of
substituting a default.

Without a grammar the schema `enum` no longer constrains the label vocabulary, and the
distillers match exact code strings (`build_nto_rate_features.FAMILIES`,
`annotate_student_states.CODEBOOK`), so an unrecognised code becomes an all-zero column in the
feature block. `VocabAudit` reports such codes.

    from anno_json import make_sampling_params, extract_json, VocabAudit
"""
from __future__ import annotations

import json
import os
import re

_FENCE_OPEN = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE = re.compile(r"\s*```\s*$")


def make_sampling_params(schema, max_tokens: int):
    """Build greedy vLLM sampling parameters, with or without a JSON grammar.

    With ``SO_OFF=1`` in the environment the result is plain greedy sampling with no grammar.
    Otherwise decoding is constrained to ``schema``; the backend is chosen at engine level by
    ``SO_BACKEND`` (see ``engine_kwargs``).

    Args:
        schema: JSON schema (dict) the output must follow when constrained decoding is on.
        max_tokens: generation budget per prompt.

    Returns:
        A ``vllm.SamplingParams`` with temperature 0.
    """
    from vllm import SamplingParams

    if os.environ.get("SO_OFF") == "1":
        return SamplingParams(temperature=0.0, max_tokens=max_tokens)
    from vllm.sampling_params import StructuredOutputsParams
    return SamplingParams(temperature=0.0, max_tokens=max_tokens,
                          structured_outputs=StructuredOutputsParams(json=schema))


def engine_kwargs():
    """Extra ``vllm.LLM(**kwargs)`` settings read from the environment.

    Environment variables:
        PP: pipeline-parallel size (default 1).
        PREFIX_CACHING: ``1`` (default) enables prefix caching.
        MAX_NUM_SEQS: cap on concurrent sequences (vLLM default when unset).
        SO_BACKEND: structured-output backend name; ignored when ``SO_OFF=1``.

    Prefix caching is on by default: every chunk sends the same system prompt (the codebook,
    ~758 tokens) in front of the dialogue, so with caching that prefix is not prefilled again
    for every chunk. The dtype passed to ``vllm.LLM`` must be "auto" for an FP8 checkpoint;
    forcing bfloat16 dequantises it.

    Returns:
        A dict of keyword arguments to pass to ``vllm.LLM``.
    """
    kw = {}
    pp = int(os.environ.get("PP", "1"))
    if pp > 1:
        kw["pipeline_parallel_size"] = pp
    # on by default: the shared codebook prefix is identical across every chunk
    if os.environ.get("PREFIX_CACHING", "1") == "1":
        kw["enable_prefix_caching"] = True
    if os.environ.get("MAX_NUM_SEQS"):
        kw["max_num_seqs"] = int(os.environ["MAX_NUM_SEQS"])
    name = os.environ.get("SO_BACKEND")
    if name and os.environ.get("SO_OFF") != "1":
        try:
            from vllm.config import StructuredOutputsConfig
            kw["structured_outputs_config"] = StructuredOutputsConfig(backend=name)
        except Exception:
            kw["guided_decoding_backend"] = name
    return kw


def extract_json(text: str):
    """Return the first parseable JSON object in a model reply.

    Tolerates what an unconstrained model emits: markdown fences, a prose preamble and
    trailing commentary. It scans for a balanced brace span instead of regex matching, and is
    string-aware, so a `}` inside a quoted value does not close the object early.

    Args:
        text: raw generated text.

    Returns:
        The parsed object, or None if nothing parses. Truncated output (unbalanced braces)
        also gives None instead of a guess: a truncated chunk has lost its last turns, which
        is a position bias, and every feature block carries `_late` features.
    """
    if not text:
        return None
    t = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip()))
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None            # unbalanced => truncated


class VocabAudit:
    """Count codes the model emitted that the distiller does not know about.

    Without the schema `enum` nothing constrains the label strings, and the distillers match
    codes exactly. An unknown code raises no error downstream; it never matches a family, so
    that feature column is all zero. This class counts such codes, and `report()` returns a
    non-zero value so a caller can reject the run.

    Args:
        allowed: iterable of valid code strings.
    """

    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.seen = 0
        self.bad: dict[str, int] = {}

    def check(self, codes):
        """Filter codes to the known vocabulary and record the ones dropped.

        Args:
            codes: code strings emitted for one turn.

        Returns:
            The codes that are in the allowed vocabulary, in their original order.
        """
        keep = []
        for c in codes:
            self.seen += 1
            if c in self.allowed:
                keep.append(c)
            else:
                self.bad[c] = self.bad.get(c, 0) + 1
        return keep

    #: Fail above this out-of-vocabulary rate, not on any single stray code: a few near-miss
    #: paraphrases of real codes are noise. Systematic drift must still abort: one renamed code
    #: appearing thousands of times is what empties a feature column.
    MAX_OOV_RATE = 0.005

    def report(self, label: str = "") -> int:
        """Print the audit summary.

        Strays are dropped, never mapped onto a real code. A stray may, for example, confuse
        `PROMPTING_*` and `PROBING_*`, two real prefixes on either side of the elicitation
        distinction. Mapping by hand across that boundary would contaminate that contrast;
        dropping a few stray codes does not.

        Args:
            label: text appended to the ``VOCAB_AUDIT`` line, e.g. the shard name.

        Returns:
            0 if the out-of-vocabulary rate is within ``MAX_OOV_RATE``, else the number of
            dropped codes.
        """
        n_bad = sum(self.bad.values())
        rate = n_bad / max(1, self.seen)
        ok = rate <= self.MAX_OOV_RATE
        print(f"VOCAB_AUDIT{label}: {self.seen:,} codes emitted, {n_bad:,} out-of-vocabulary "
              f"({rate:.3%}), {'within' if ok else 'above'} the {self.MAX_OOV_RATE:.1%} "
              f"tolerance", flush=True)
        if self.bad:
            top = sorted(self.bad.items(), key=lambda kv: -kv[1])[:8]
            print("  dropped codes: " + ", ".join(f"{k!r}x{v}" for k, v in top), flush=True)
        if not ok:
            print("  STOP: systematic vocabulary drift. An unknown code becomes an all-zero "
                  "feature column downstream, because the distillers match code strings exactly. "
                  "Fix the prompt or the codebook before distilling.", flush=True)
        return 0 if ok else n_bad
