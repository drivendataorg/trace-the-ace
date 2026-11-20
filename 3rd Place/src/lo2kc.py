"""LO -> KC prediction for unseen objectives: embedding match to CCSS.

At test time the learning objective is unseen and there is no API, so a
shipped component must predict its KCs. Baseline here: MiniLM-embed the LO
text and every CCSS standard description (K-8), take standards above a
cosine threshold relative to the best match (plus top-k cap). Zero training
data -> no leakage, generalizes to any curriculum phrasing.

Writes data/lo_kcs_pred.json for the mastery features without requiring
manual objective annotations.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from training_paths import DATA
EMB = str(Path(__file__).resolve().parents[1] / "models/minilm")
TOPK = 4
REL = 0.85          # keep standards with cosine >= REL * best


def embed(texts, tok, model, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=96)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.vstack(out)


def main():
    h = json.load(open(DATA / "ccss_hier.json"))
    import re
    std_re = re.compile(r"^(K|\d)\.([A-Z]+)\.([A-Z])\.(\d+)$")
    codes, descs = [], []
    for k, v in h.items():
        if std_re.match(k):
            codes.append(k)
            descs.append(v)
    print(f"{len(codes)} CCSS standards")

    los = {}
    for r in csv.DictReader(open(DATA / "train_features_TMQTWsB.csv")):
        los.setdefault(r["learning_objective_id"],
                       r["learning_objective"].strip())
    lo_ids = sorted(los)

    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB).eval()
    E_std = embed(descs, tok, model)
    E_lo = embed([los[i] for i in lo_ids], tok, model)
    S = E_lo @ E_std.T                        # (398, n_std)

    pred = {}
    for i, oid in enumerate(lo_ids):
        s = S[i]
        top = np.argsort(-s)[:TOPK]
        best = s[top[0]]
        keep = [codes[j] for j in top if s[j] >= REL * best]
        pred[oid] = keep
    json.dump(pred, open(DATA / "lo_kcs_pred.json", "w"))


    print("wrote lo_kcs_pred.json")


if __name__ == "__main__":
    sys.exit(main())
