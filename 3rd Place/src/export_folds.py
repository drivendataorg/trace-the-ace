"""Export the exact fold assignments used by hardened_model.py so the LLMKT
run on trains/evaluates on identical double-blocked splits.

Writes data/folds.csv: response_id, session_id, learning_objective_id,
obj_fold, clus_fold, y.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from training_paths import DATA

npz = np.load(DATA / "hardened.npz", allow_pickle=False)
sessions = list(npz["sessions"])
s_idx = {s: i for i, s in enumerate(sessions)}

labels = {r["response_id"]: float(r["is_correct"])
          for r in csv.DictReader(open(DATA / "train_labels_44ujmj2.csv"))}
rows = [r for r in csv.DictReader(open(DATA / "train_features_TMQTWsB.csv"))
        if r["response_id"] in labels and r["session_id"] in s_idx]
y = np.array([labels[r["response_id"]] for r in rows])
obj = np.array([r["learning_objective_id"] for r in rows])
ri = np.array([s_idx[r["session_id"]] for r in rows])

H = sparse.load_npz(DATA / "hardened_hash.npz")
V = TruncatedSVD(min(100, H.shape[0] - 1, H.shape[1] - 1), random_state=0).fit_transform(normalize(H))
km = KMeans(min(10, len(sessions)), n_init=10, random_state=0).fit(normalize(V))
clus_fold = (km.labels_ // 2)[ri]

obj_fold = np.zeros(len(rows), int)
for f, (_, te) in enumerate(
        GroupKFold(min(5, len(set(obj)))).split(np.zeros(len(rows)), y, obj)):
    obj_fold[te] = f

with open(DATA / "folds.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["response_id", "session_id", "learning_objective_id",
                "obj_fold", "clus_fold", "y"])
    for i, r in enumerate(rows):
        w.writerow([r["response_id"], r["session_id"],
                    r["learning_objective_id"], obj_fold[i], clus_fold[i],
                    int(y[i])])
print(len(rows), "rows written")
print("train pool (of!=0 & cf!=0):",
      int(np.sum((obj_fold != 0) & (clus_fold != 0))))
print("double-blocked eval (of==0 & cf==0):",
      int(np.sum((obj_fold == 0) & (clus_fold == 0))))
