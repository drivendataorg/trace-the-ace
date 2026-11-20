#!/usr/bin/env python
"""Build TRACEACE_PKL-compatible MRBench tables for the stage-2 LoRA.

MRBench is CC BY-SA 4.0; these tables and the adapters trained on them are released under
CC BY-SA 4.0 (see data/external/mrbench/NOTICE.md and models/llm/NOTICE.md at the repository
root).

Labels are the Qwen3-32B pseudo-labels from annotate_mrbench_vllm.py. Every labelled turn gets
the generic KC "solving a math word problem", because MRBench has no topic field (a known
limitation). The split is 80/20 by conversation_id with seed 221.

Output: data/annotated/mrbench_train.pkl and data/annotated/mrbench_test.pkl.
"""
import os, random, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_mrbench_vllm import load_mrbench_dialogues

KC = "solving a math word problem"


def main():
    """Write the MRBench train and test pickles from the per-turn label CSV.

    Environment variables:
        MRB_LABELS: label CSV (default results/mrbench_turn_labels_qwen32b.csv).
        MRB_SUFFIX: suffix for the output file names (default empty).

    Each dialogue with at least one true/false label becomes one row (index, dialogue,
    annotation, meta_data, lo_id); unlabelled turns keep ``correct=None``.

    Returns:
        None. Writes data/annotated/mrbench_{train,test}<MRB_SUFFIX>.pkl.
    """
    # MRB_LABELS / MRB_SUFFIX: the defaults read the Qwen3-32B labels and write the
    # unsuffixed pickles.
    tl = pd.read_csv(os.environ.get("MRB_LABELS", "results/mrbench_turn_labels_qwen32b.csv"))
    tl["dialogue_id"] = tl["dialogue_id"].astype(str)
    tl = tl[tl.label.isin(["true", "false"])]
    labels = {d: list(zip(g.turn, g.label == "true")) for d, g in tl.groupby("dialogue_id")}
    print(f"labels: {sum(len(v) for v in labels.values())} true/false over {len(labels)} dialogues")

    ds = load_mrbench_dialogues()
    cids = sorted(d[0] for d in ds)
    rng = random.Random(221)
    rng.shuffle(cids)
    test_ids = set(cids[: len(cids) // 5])

    rows = {"train": [], "test": []}
    for cid, turns in ds:
        cand = [(t, c) for t, c in labels.get(cid, []) if 1 <= t <= len(turns)]
        if not cand:
            continue
        real = [dict(t) for t in turns]
        for i, t in enumerate(real, start=1):
            t["turn"] = i
        anno = {f"turn {t['turn']}": {"correct": None, "kcs": []} for t in real}
        for t, c in cand:
            anno[f"turn {t}"] = {"correct": bool(c), "kcs": [KC]}
        rows["test" if cid in test_ids else "train"].append({
            "index": cid, "dialogue": real, "annotation": anno,
            "meta_data": {"conversation_id": cid, "mrbench_split": "test" if cid in test_ids else "train"},
            "lo_id": KC,
        })
    for name, rs in rows.items():
        out = pd.DataFrame(rs)
        out.to_pickle(f"data/annotated/mrbench_{name}{os.environ.get('MRB_SUFFIX', '')}.pkl")
        lab = [a["correct"] for r in rs for a in r["annotation"].values() if a["correct"] is not None]
        print(f"mrbench_{name}.pkl: {len(out)} dialogues / {len(lab)} labeled / "
              f"{pd.Series(lab).mean():.3f} true")


if __name__ == "__main__":
    main()
