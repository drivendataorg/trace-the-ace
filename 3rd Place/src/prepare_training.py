"""Validate raw data and prepare/validate human annotation sheets."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict

from training_paths import DATA, ROOT, TRANSCRIPTS


def read_csv(path, columns):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def raw_fingerprint():
    features = read_csv(DATA / "train_features_TMQTWsB.csv",
                        ("response_id", "session_id", "learning_objective_id", "learning_objective"))
    labels = read_csv(DATA / "train_labels_44ujmj2.csv", ("response_id", "is_correct"))
    if not features or not labels:
        raise ValueError("Training features and labels must not be empty")
    for rows, name in ((features, "features"), (labels, "labels")):
        ids = [r["response_id"] for r in rows]
        if len(set(ids)) != len(ids) or any(not x for x in ids):
            raise ValueError(f"Duplicate or empty response_id in {name}")
    if {r["response_id"] for r in features} != {r["response_id"] for r in labels}:
        raise ValueError("Feature and label response IDs must match exactly")
    if any(r["is_correct"] not in ("0", "1", "0.0", "1.0") for r in labels):
        raise ValueError("is_correct must be 0 or 1")
    if len({float(r["is_correct"]) for r in labels}) != 2:
        raise ValueError("Training requires both correct and incorrect quiz outcomes")
    if len({r["session_id"] for r in features}) < 2 or len({r["learning_objective_id"] for r in features}) < 2:
        raise ValueError("Training requires at least two sessions and two learning objectives")
    digest = hashlib.sha256()
    for filename in ("train_features_TMQTWsB.csv", "train_labels_44ujmj2.csv"):
        digest.update((DATA / filename).read_bytes())
    for session in sorted({r["session_id"] for r in features}):
        if not session or "/" in session or "\\" in session or session in (".", ".."):
            raise ValueError(f"Invalid session_id: {session!r}")
        path = TRANSCRIPTS / f"{session}.csv"
        # Content hashes ensure cached intermediates cannot outlive edited transcripts.
        digest.update(session.encode())
        digest.update(path.read_bytes())
        with open(path, encoding="utf-8-sig", newline="") as f:
            fields = csv.DictReader(f).fieldnames or []
        if not {"role", "content"} <= set(fields):
            raise ValueError(f"{path}: transcript needs role and content columns")
    return digest.hexdigest()


def judge_sheet(limit=1000):
    """Only sample the original training pool; humans supply the verdicts."""
    folds = read_csv(DATA / "folds.csv", ("session_id", "obj_fold", "clus_fold"))
    sessions = sorted({r["session_id"] for r in folds
                       if r["obj_fold"] != "0" and r["clus_fold"] != "0"})
    if not sessions:
        raise ValueError("No sessions available for human judge annotations")
    picked = set(random.Random(0).sample(sessions, min(limit, len(sessions))))
    path = DATA / "turn_labels.csv"
    if path.exists():
        print(f"Preserving human annotation sheet: {path}")
        return
    history = defaultdict(list)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session_id", "pair_idx", "context",
                                               "tutor", "student", "verdict"])
        writer.writeheader()
        for line in open(DATA / "pairs.jsonl"):
            p = json.loads(line)
            session = p["session_id"]
            if session not in picked:
                continue
            ctx = history[session][-3:]
            writer.writerow({"session_id": session, "pair_idx": p["pair_idx"],
                             "context": "\n".join(ctx), "tutor": p["tutor"],
                             "student": p["student"], "verdict": ""})
            history[session].append(f"Tutor: {p['tutor']}\nStudent: {p['student']}")
    print(f"Fill the verdict column in {path}: true, false, or na")


def cluster_sheet():
    path = DATA / "cluster_labels.csv"
    if path.exists():
        print(f"Preserving human annotation sheet: {path}")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster", "examples", "kc_codes"])
        writer.writeheader()
        for line in open(DATA / "exemplars.jsonl"):
            row = json.loads(line)
            writer.writerow({"cluster": row["cluster"],
                             "examples": "\n\n".join(row["exemplars"]), "kc_codes": ""})
    print(f"Fill kc_codes in {path} using ccss_standards.csv; use [] for no matching skill")


def reference():
    source = ROOT / "src/resources/ccss_hier.json"
    shutil.copyfile(source, DATA / "ccss_hier.json")
    import re
    standards = {k: v for k, v in json.loads(source.read_text()).items()
                 if re.fullmatch(r"(K|\d)\.[A-Z]+\.[A-Z]\.\d+", k)}
    with open(DATA / "ccss_standards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "description"])
        writer.writerows(sorted(standards.items()))


def validated_verdicts():
    rows = read_csv(DATA / "turn_labels.csv", ("session_id", "pair_idx", "verdict"))
    labels = {}
    for r in rows:
        verdict = r["verdict"].strip().lower()
        if not verdict:
            continue  # Partial human annotation is allowed.
        key = (r["session_id"], int(r["pair_idx"]))
        if verdict not in ("true", "false", "na"):
            raise ValueError(f"Invalid verdict for {key}: {verdict!r}")
        if key in labels:
            raise ValueError(f"Duplicate human annotation for {key}")
        labels[key] = verdict
    if len({s for s, _ in labels}) < 2:
        raise ValueError("Fill turn_labels.csv with human verdicts from at least two sessions")
    remaining = set(labels)
    for line in open(DATA / "pairs.jsonl"):
        p = json.loads(line)
        remaining.discard((p["session_id"], p["pair_idx"]))
    if remaining:
        raise ValueError(f"Annotations reference unknown pairs: {sorted(remaining)[:5]}")
    return labels


def validate_clusters():
    standards = {r["code"] for r in read_csv(DATA / "ccss_standards.csv", ("code",))}
    expected = {str(json.loads(line)["cluster"]) for line in open(DATA / "exemplars.jsonl")}
    labels = {}
    for r in read_csv(DATA / "cluster_labels.csv", ("cluster", "kc_codes")):
        cluster = r["cluster"]
        if cluster not in expected or cluster in labels:
            raise ValueError(f"Unknown or duplicate cluster: {cluster}")
        raw = r["kc_codes"].strip()
        if not raw:
            raise ValueError(f"Fill kc_codes for cluster {cluster}; use [] for no matching skill")
        codes = json.loads(raw) if raw.startswith("[") else raw.replace(";", ",").split(",")
        if not isinstance(codes, list) or any(not isinstance(c, str) for c in codes):
            raise ValueError(f"Cluster {cluster}: kc_codes must be a list of CCSS codes")
        codes = [c.strip() for c in codes]
        if set(codes) - standards:
            raise ValueError(f"Cluster {cluster}: unknown CCSS codes {set(codes) - standards}")
        labels[cluster] = codes
    if set(labels) != expected:
        raise ValueError("cluster_labels.csv must contain every generated cluster")
    if not any(labels.values()):
        raise ValueError("At least one cluster must have a CCSS skill annotation")
    (DATA / "kc_labels.json").write_text(json.dumps({f"cluster-{k}": v for k, v in labels.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("judge-sheet", "cluster-sheet", "reference",
                                            "validate-judge", "validate-clusters"))
    args = parser.parse_args()
    {"judge-sheet": judge_sheet, "cluster-sheet": cluster_sheet, "reference": reference,
     "validate-judge": validated_verdicts, "validate-clusters": validate_clusters}[args.action]()


if __name__ == "__main__":
    main()
