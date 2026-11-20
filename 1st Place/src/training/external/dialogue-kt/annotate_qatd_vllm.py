"""QATD-2k per-turn correctness annotation with Qwen3-32B via vLLM.

QATD-2k (Eedi/Question-Anchored-Tutoring-Dialogues-2k, CC BY-NC 4.0, so evaluation only and
never used for shipped weights) is real Eedi tutor-student chat. It has no correctness labels,
so we pseudo-label each turn with Qwen3-32B, using the prompts and chunking of
annotate_traceace_qwen32b.py / annotate_traceace_vllm.py. annotate_mrbench_vllm.py (step 02f)
imports `chunk_prompts` from here.

Output: results/qatd_turn_labels_qwen32b.csv (intervention_id, turn, label). `turn` is the
1-based position in the process_dialogue() turn list, the same convention as the competition
labels CSV. Unparseable replies are logged to results/qatd_anno_parse_fails.jsonl.

Environment: QATD_DIR (the dataset folder; the default is the path on our cluster, so set it),
ANNO_MODEL (Qwen/Qwen3-32B), ANNO_TP=1, ANNO_PP=1, ANNO_CHUNK_TURNS=30, ANNO_MAXTOK=2048,
ANNO_MAXLEN=8192, ANNO_MAX_SEQS=512, ANNO_N_DIALOGUES (0 = all), ANNO_OUT. Resume-safe: skips
intervention_ids already in the output CSV.
"""
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.annotate import extract_result
from dialogue_kt.prompting import ANNO_CORRECTNESS_SYSTEM_PROMPT, TRACEACE_DIALOGUE_DESC
from dialogue_kt.data_loading import process_dialogue

QATD = os.environ.get("QATD_DIR", "/dtu-compute/sciofsci/magnificat/external/datasets/qatd2k")
MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
TP = int(os.environ.get("ANNO_TP", "1"))
CHUNK = int(os.environ.get("ANNO_CHUNK_TURNS", "30"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "2048"))
N_DIALOGUES = int(os.environ.get("ANNO_N_DIALOGUES", "0"))  # 0 = all
OUT = os.environ.get("ANNO_OUT", "results/qatd_turn_labels_qwen32b.csv")
BLOCK = 400  # dialogues per generate() call (QATD dialogues are short; ~1-2 chunks each)


def load_qatd_dialogues():
    """Load the QATD-2k train and test dialogues as dialogue-kt turn lists.

    Returns:
        A list of ``(intervention_id, turns)`` pairs, where ``turns`` is the output of
        ``dialogue_kt.data_loading.process_dialogue`` (empty messages dropped).
    """
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(f"{QATD}/anchored-dialogues/{split}.csv")
        frames.append(df)
    msgs = pd.concat(frames, ignore_index=True)
    out = []
    for iid, g in msgs.groupby("InterventionId"):
        g = g.sort_values("MessageSequence")
        utts = []
        for _, m in g.iterrows():
            content = "" if pd.isna(m["MessageString"]) else str(m["MessageString"]).strip()
            if not content:
                continue
            role = "teacher" if int(m["IsTutor"]) == 1 else "student"
            utts.append({"role": role, "content": content})
        if not utts:
            continue
        turns = process_dialogue(utts)
        if turns:
            out.append((int(iid), turns))
    return out


def chunk_prompts(iid: int, turns: list):
    """Split a dialogue into user prompts of at most CHUNK turns.

    Same chunking and prompt text as ``annotate_traceace_qwen32b.chunk_prompts``.

    Args:
        iid: intervention id of the dialogue.
        turns: the dialogue's turn list.

    Yields:
        ``((iid, start, k), prompt)`` with ``start`` the 0-based offset of the chunk and ``k``
        its number of turns (numbered 1..k in the prompt).
    """
    for start in range(0, len(turns), CHUNK):
        chunk = turns[start:start + CHUNK]
        lines = []
        for i, t in enumerate(chunk, start=1):
            if t["teacher"]:
                lines.append(f"Teacher Turn {i}: {t['teacher']}")
            if t["student"]:
                lines.append(f"Student Turn {i}: {t['student']}")
        prompt = "[BEGIN DIALOGUE]\n" + "\n".join(lines) + "\n[END DIALOGUE]"
        prompt += f"\n\nYour final response should have an entry for exactly {len(chunk)} turns (1-{len(chunk)})."
        yield (iid, start, len(chunk)), prompt


def main():
    """Annotate the QATD-2k dialogues not yet in OUT and append the labels to it.

    Returns:
        None. Appends ``intervention_id, turn, label`` rows to OUT.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    dialogues = load_qatd_dialogues()
    print(f"QATD dialogues: {len(dialogues)}  "
          f"turns median {pd.Series([len(t) for _, t in dialogues]).median()}", flush=True)
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["intervention_id"].unique())
        print(f"resume: {len(done)} dialogues already annotated", flush=True)
    dialogues = [(i, t) for i, t in dialogues if i not in done]
    if N_DIALOGUES:
        dialogues = dialogues[:N_DIALOGUES]
    print(f"to annotate: {len(dialogues)}", flush=True)

    system = ANNO_CORRECTNESS_SYSTEM_PROMPT.format(desc=TRACEACE_DIALOGUE_DESC)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # max_num_seqs: Qwen3.5-122B is a hybrid Mamba model and needs one Mamba cache
    # block per decode sequence. vLLM's default (1024) exceeds the blocks available at
    # this gpu_memory_utilization, so CUDA-graph capture aborts engine startup with
    # "max_num_seqs exceeds available Mamba cache blocks". Harmless for the 32B default.
    # ANNO_PP: pipeline-parallel size (default 1); optional.
    llm = LLM(model=MODEL, dtype="auto", tensor_parallel_size=TP,
              pipeline_parallel_size=int(os.environ.get("ANNO_PP", "1")),
              max_model_len=int(os.environ.get("ANNO_MAXLEN", "8192")),
              gpu_memory_utilization=0.92, enforce_eager=False,
              max_num_seqs=int(os.environ.get("ANNO_MAX_SEQS", "512")))
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    os.makedirs("results", exist_ok=True)
    header_needed = not os.path.exists(OUT)
    n_parse_fail, n_chunks = 0, 0

    for bstart in range(0, len(dialogues), BLOCK):
        block = dialogues[bstart:bstart + BLOCK]
        metas, prompts = [], []
        for iid, turns in block:
            for meta, up in chunk_prompts(iid, turns):
                metas.append(meta)
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": up}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
        n_chunks += len(prompts)
        outs = llm.generate(prompts, sp)
        rows = []
        for (iid, start, k), o in zip(metas, outs):
            pred = extract_result(o.outputs[0].text)
            if not isinstance(pred, dict):
                n_parse_fail += 1
                import json as _json
                with open("results/qatd_anno_parse_fails.jsonl", "a") as fh:
                    fh.write(_json.dumps({
                        "intervention_id": iid, "start": start, "k": k,
                        "finish_reason": str(o.outputs[0].finish_reason),
                        "n_out_tokens": len(o.outputs[0].token_ids),
                        "tail": o.outputs[0].text[-400:]}) + "\n")
                continue
            for key, val in pred.items():
                try:
                    local = int(str(key).lower().replace("turn", "").strip())
                except ValueError:
                    continue
                if 1 <= local <= k and str(val).lower().strip() in ("true", "false", "na"):
                    rows.append({"intervention_id": iid, "turn": start + local,
                                 "label": str(val).lower().strip()})
        if rows:
            pd.DataFrame(rows).to_csv(OUT, mode="a", header=header_needed, index=False)
            header_needed = False
        print(f"  dialogues {min(bstart + BLOCK, len(dialogues))}/{len(dialogues)}  "
              f"chunks {n_chunks}  parse_fail {n_parse_fail}", flush=True)

    print(f"DONE: chunks {n_chunks}, parse_fail {n_parse_fail} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
