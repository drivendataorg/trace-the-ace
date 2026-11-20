"""mas7 = the 6 shipped mastery features + log1p(n_rel): number of assessable
pairs in the session whose KC cluster matches the objective's KCs at ANY
hierarchy level (a coverage feature -- 'how much was this objective
exercised').  Same ingredients as the shipped mastery features (LO->CCSS
mapping + pair clustering); no LO identity."""
import numpy as np
from kc_fulltest import mastery_features, kc_levels


def nrel_feature(rows, probs, assign, clus_kcs, lo_kcs):
    sess_lv = {}
    for s in {r["session_id"] for r in rows}:
        lvs = []
        for i, (pt, pf, pn) in probs.get(s, {}).items():
            c = assign.get(s, {}).get(i)
            if c is None or c not in clus_kcs:
                continue
            lv = [set(), set(), set()]
            for code in clus_kcs[c]:
                for L, part in enumerate(kc_levels(code)):
                    lv[L].add(part)
            lvs.append(lv)
        sess_lv[s] = lvs
    out = np.zeros(len(rows))
    for k, r in enumerate(rows):
        tsets = [set(), set(), set()]
        for code in lo_kcs.get(r["learning_objective_id"], []):
            for L, part in enumerate(kc_levels(code)):
                tsets[L].add(part)
        n = sum(1 for lv in sess_lv[r["session_id"]]
                if any(lv[L] & tsets[L] for L in range(3) if tsets[L]))
        out[k] = np.log1p(n)
    return out


def mastery_features7(rows, probs, assign, clus_kcs, lo_kcs):
    M6, lev = mastery_features(rows, probs, assign, clus_kcs, lo_kcs)
    return np.hstack([M6, nrel_feature(rows, probs, assign, clus_kcs, lo_kcs)[:, None]]), lev
