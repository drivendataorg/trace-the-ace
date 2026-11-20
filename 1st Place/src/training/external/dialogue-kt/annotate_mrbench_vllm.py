"""MRBench per-turn correctness annotation with Qwen3-32B via vLLM.

MRBench (MBZUAI; CC BY-SA 4.0, and the labels and adapters derived from it are released under
CC BY-SA 4.0, see data/external/mrbench/NOTICE.md at the repository root) has ~883 unique
conversation contexts (derived from MathDial and Bridge, GSM8k-like problems). The JSON files
are read from external/datasets/mrbench/ in the workspace (src/make_workspace.py copies them
there from data/external/mrbench/raw/); set MRB_DIR to read them from elsewhere.

MRBench has no correctness labels and no topic or KC field, so we pseudo-label each turn with
Qwen3-32B, using the dialogue-kt correctness prompt and the chunking of annotate_qatd_vllm.py
(`chunk_prompts`). At build time the KC is a generic phrase ("solving a math word problem"):
there is no per-dialogue objective to condition on. These labels are the training targets of
the stage-2 LoRA.

Environment: ANNO_MODEL (Qwen/Qwen3-32B), ANNO_TP=1, ANNO_PP=1, ANNO_MAXTOK=2048,
ANNO_MAXLEN=8192, ANNO_MAX_SEQS=512, ANNO_THINK (off), ANNO_OUT, MRB_DIR.

Output: results/mrbench_turn_labels_qwen32b.csv (dialogue_id, turn, label). Resume-safe.
"""
import glob, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.annotate import extract_result
from dialogue_kt.prompting import ANNO_CORRECTNESS_SYSTEM_PROMPT, TRACEACE_DIALOGUE_DESC
from dialogue_kt.data_loading import process_dialogue
from annotate_qatd_vllm import chunk_prompts

MRB = os.environ.get("MRB_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "datasets", "mrbench"))  # repo-relative
MODEL = os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B")
TP = int(os.environ.get("ANNO_TP", "1"))
MAXTOK = int(os.environ.get("ANNO_MAXTOK", "2048"))
OUT = os.environ.get("ANNO_OUT", "results/mrbench_turn_labels_qwen32b.csv")


def load_mrbench_dialogues():
    """Load the MRBench conversation histories as dialogue-kt turn lists.

    Each line of a ``conversation_history`` is split on its first colon; lines starting with
    "student" or "pupil" are student utterances, all others tutor utterances.

    Returns:
        A list of ``(conversation_id, turns)`` pairs, one per conversation id across all
        ``MRB_DIR/*.json`` files (first occurrence kept), where ``turns`` is the output of
        ``dialogue_kt.data_loading.process_dialogue``.
    """
    seen, out = set(), []
    for f in sorted(glob.glob(f"{MRB}/*.json")):
        for r in json.load(open(f)):
            cid = str(r.get("conversation_id"))
            ch = r.get("conversation_history")
            if cid in seen or not isinstance(ch, str):
                continue
            seen.add(cid)
            utts = []
            for line in ch.split("\n"):
                line = line.strip()
                if not line:
                    continue
                who = "student" if line.lower().startswith(("student", "pupil")) else "teacher"
                body = line.split(":", 1)[1].strip() if ":" in line else line
                if body:
                    utts.append({"role": who, "content": body})
            if not utts:
                continue
            turns = process_dialogue(utts)
            if turns:
                out.append((cid, turns))
    return out


def main():
    """Annotate every MRBench dialogue not yet in ``OUT`` and append the labels to it.

    Returns:
        None. Appends ``dialogue_id, turn, label`` rows (label in true/false/na) to ``OUT``.
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    ds = load_mrbench_dialogues()
    print(f"MRBench unique dialogues: {len(ds)}", flush=True)
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["dialogue_id"].astype(str).unique())
    ds = [d for d in ds if d[0] not in done]
    print(f"to annotate: {len(ds)}", flush=True)
    if not ds:
        return

    system = ANNO_CORRECTNESS_SYSTEM_PROMPT.format(desc=TRACEACE_DIALOGUE_DESC)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # max_num_seqs: Qwen3.5-122B is a hybrid Mamba model and needs one Mamba cache
    # block per decode sequence. vLLM's default (1024) exceeds the blocks available at
    # this gpu_memory_utilization, so CUDA-graph capture aborts engine startup with
    # "max_num_seqs exceeds available Mamba cache blocks". Harmless for the 32B default.
    # ANNO_THINK=1 enables reasoning (off by default).
    # ANNO_PP: pipeline-parallel size (default 1).
    llm = LLM(model=MODEL, dtype="auto", tensor_parallel_size=TP,
              pipeline_parallel_size=int(os.environ.get("ANNO_PP", "1")),
              max_model_len=int(os.environ.get("ANNO_MAXLEN", "8192")),
              gpu_memory_utilization=0.92, enforce_eager=False,
              max_num_seqs=int(os.environ.get("ANNO_MAX_SEQS", "512")))
    sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK)

    metas, prompts = [], []
    for cid, turns in ds:
        for meta, up in chunk_prompts(cid, turns):
            metas.append(meta)
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": up}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=os.environ.get("ANNO_THINK") == "1"))
    outs = llm.generate(prompts, sp)
    rows, n_fail = [], 0
    for (cid, start, k), o in zip(metas, outs):
        pred = extract_result(o.outputs[0].text)
        if not isinstance(pred, dict):
            n_fail += 1
            continue
        for key, val in pred.items():
            try:
                local = int(str(key).lower().replace("turn", "").strip())
            except ValueError:
                continue
            if 1 <= local <= k and str(val).lower().strip() in ("true", "false", "na"):
                rows.append({"dialogue_id": cid, "turn": start + local,
                             "label": str(val).lower().strip()})
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print(f"DONE: chunks {len(prompts)}, parse_fail {n_fail} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
