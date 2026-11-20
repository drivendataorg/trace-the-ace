"""Stage A of the KC pipeline: embed assessable QA pairs and cluster them.

Assessable = student turn has digits OR >= 8 words (48.7% of 2.37M pairs).
Each pair is embedded as "T: <tutor> S: <student>" with MiniLM mean pooling,
then MiniBatchKMeans into K clusters. Clusters are the unit of KC labeling
(humans label each cluster once via exemplars) and of inference-time KC
assignment (nearest cluster -- no LLM call needed per pair).

Outputs (experiment5/kc/):
  assign.csv       session_id,pair_idx,cluster
  centers.npy      K x 384 float32 (L2-normalised space)
  exemplars.jsonl  per cluster: 5 pairs nearest the center, with texts
Run on kookree: venv python, A100.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
from training_paths import DATA
OUT = DATA
EMB_MODEL = str(BASE / "models/minilm")
K = 2000
BATCH = 1024
MAXTOK = 96


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    meta, texts = [], []
    for line in open(DATA / "pairs.jsonl"):
        p = json.loads(line)
        if not (p["stu_has_digits"] or p["stu_words"] >= 8):
            continue
        meta.append((p["session_id"], p["pair_idx"]))
        texts.append(f"T: {p['tutor'][:400]} S: {p['student'][:300]}")
    n = len(texts)
    print(f"{n} assessable pairs  [{time.time()-t0:.0f}s]", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(
        EMB_MODEL, dtype=torch.float32).to(device).eval()
    E = np.empty((n, 384), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, n, BATCH):
            enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAXTOK).to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            e = (h * m).sum(1) / m.sum(1)
            e = torch.nn.functional.normalize(e, dim=-1)
            E[i:i + BATCH] = e.cpu().numpy().astype(np.float16)
            if (i // BATCH) % 200 == 0:
                print(f"  embed {i}/{n}  [{time.time()-t0:.0f}s]", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"embedded  [{time.time()-t0:.0f}s]", flush=True)

    n_clusters = min(K, n)
    if n_clusters == 0:
        raise ValueError("No assessable pairs found in the transcripts")
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=20_000, n_init=3,
                         random_state=0, max_iter=200)
    CH = 200_000
    for ep in range(3):
        for i in range(0, n, CH):
            km.partial_fit(E[i:i + CH].astype(np.float32))
        print(f"kmeans epoch {ep}  [{time.time()-t0:.0f}s]", flush=True)
    lab = np.empty(n, dtype=np.int32)
    dist = np.empty(n, dtype=np.float32)
    for i in range(0, n, CH):
        X = E[i:i + CH].astype(np.float32)
        similarity = X @ km.cluster_centers_.T
        lab[i:i + CH] = similarity.argmax(1)
        dist[i:i + CH] = -similarity.max(1)
    np.save(OUT / "centers.npy", km.cluster_centers_.astype(np.float32))

    with open(OUT / "assign.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "pair_idx", "cluster"])
        for (s, pi), c in zip(meta, lab):
            w.writerow([s, pi, int(c)])

    with open(OUT / "exemplars.jsonl", "w") as f:
        order = np.argsort(dist)
        chosen = {c: [] for c in range(n_clusters)}
        for j in order:
            c = int(lab[j])
            if len(chosen[c]) < 5:
                chosen[c].append(int(j))
        sizes = np.bincount(lab, minlength=n_clusters)
        for c in range(n_clusters):
            f.write(json.dumps({
                "cluster": c, "size": int(sizes[c]),
                "exemplars": [texts[j] for j in chosen[c]]}) + "\n")
    print(f"done: {n_clusters} clusters, sizes p10 {np.percentile(sizes,10):.0f} "
          f"median {np.median(sizes):.0f} p90 {np.percentile(sizes,90):.0f} "
          f"empty {(sizes==0).sum()}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
