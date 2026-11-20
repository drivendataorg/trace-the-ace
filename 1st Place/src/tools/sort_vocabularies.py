#!/usr/bin/env python
"""Re-save model files with their TF-IDF vocabularies in sorted order, and prove nothing changed.

A fitted scikit-learn vectorizer keeps ``vocabulary_`` in the order it first met each term in the
training text, so the saved dict replays fragments of the training transcripts. This tool loads
each file, rebuilds every fitted vocabulary as the same term -> column mapping in sorted order
(``magnificat.vocab_order.resave_sorted``, the same function the training scripts save with), and
checks, for every vectorizer, that ``transform`` gives an identical sparse matrix
before and after on a probe corpus made of the vocabulary's own terms. Files are written back
with the compression they had (zlib level 3 for the distilled classifiers, none for the XGBoost
model).

    python src/tools/sort_vocabularies.py --check            # report, change nothing
    python src/tools/sort_vocabularies.py --in-place         # rewrite the default file set
    python src/tools/sort_vocabularies.py --in-place a.joblib b.joblib

Exit status is 1 if a file still has an unsorted vocabulary (--check) or a transform differs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/training/src"))           # magnificat.vocab_order
sys.path.insert(1, str(REPO / "models/submission/assets"))   # classes the XGBoost pickle needs

import joblib  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from magnificat.vocab_order import resave_sorted  # noqa: E402

DEFAULT = [
    *sorted((REPO / "models/distilled").glob("*.joblib")),
    *sorted((REPO / "models/gbm").glob("*.joblib")),
    *[REPO / "models/submission/assets" / n for n in (
        "baseline.joblib", "doc29_clf.joblib", "nto_moves_clf.joblib", "stu_states_clf.joblib",
        "turn_correctness_clf.joblib", "v2s_clf.joblib", "v2t_clf.joblib")],
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vectorizers(obj, seen=None):
    """Yield every fitted CountVectorizer (TfidfVectorizer included) reachable from ``obj``.

    Args:
        obj: a loaded model object; dicts, lists, tuples and object attributes are walked.
        seen: ids of objects already visited (internal, for the recursion).

    Returns:
        A generator of the vectorizers that have a non-empty ``vocabulary_``, in walk order.
    """
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, CountVectorizer):
        if getattr(obj, "vocabulary_", None):
            yield obj
        return
    kids = (obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple))
            else vars(obj).values() if hasattr(obj, "__dict__") and not isinstance(obj, type) else ())
    for k in kids:
        yield from vectorizers(k, seen)


def probe_docs(vec, n_docs=400, per_doc=60):
    """Build a fixed probe corpus of ``n_docs`` documents, each ``per_doc`` random vocabulary terms."""
    terms = list(vec.vocabulary_)
    rng = random.Random(0)
    return [" ".join(rng.choice(terms) for _ in range(per_doc)) for _ in range(n_docs)]


def same_matrix(a, b) -> bool:
    """True if two sparse matrices have the same shape, sparsity pattern and values."""
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices(); b.sort_indices()
    return (a.shape == b.shape and (a.indptr == b.indptr).all() and (a.indices == b.indices).all()
            and (a.data == b.data).all())


def main() -> None:
    """Check or rewrite the vocabularies of the given model files (default: the shipped set).

    Returns:
        Nothing; with ``--in-place`` rewrites each unsorted file in place (same compression,
        same file mode). Exits 1 if a file is still unsorted under ``--check`` or a transform
        differs after re-saving (the file is then left unchanged), and 0 otherwise.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="report only")
    g.add_argument("--in-place", action="store_true", help="rewrite files that are not sorted yet")
    args = ap.parse_args()

    bad = 0
    for p in (args.files or DEFAULT):
        p = p.resolve()
        obj = joblib.load(p)
        vecs = list(vectorizers(obj))
        unsorted = [v for v in vecs if list(v.vocabulary_) != sorted(v.vocabulary_)]
        rel = p.relative_to(REPO) if p.is_relative_to(REPO) else p
        if not vecs:
            print(f"  no vocabulary   {rel}")
            continue
        if not unsorted:
            print(f"  sorted          {rel}  ({len(vecs)} vocabularies)")
            continue
        if args.check:
            print(f"  UNSORTED        {rel}  ({len(unsorted)} of {len(vecs)} vocabularies)")
            bad += 1
            continue
        compress = 3 if p.read_bytes()[:1] == b"\x78" else 0
        old_sha = sha(p)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        os.close(fd)
        shutil.copyfile(p, tmp)
        resave_sorted(tmp, compress)                 # fresh process: reproducible bytes
        new_obj = joblib.load(tmp)
        n = 0
        for old, new in zip(vectorizers(obj), vectorizers(new_obj)):
            docs = probe_docs(old)
            if (old.vocabulary_ != new.vocabulary_ or list(new.vocabulary_) != sorted(new.vocabulary_)
                    or not same_matrix(old.transform(docs), new.transform(docs))):
                os.remove(tmp)
                print(f"  TRANSFORM DIFFERS {rel}")
                sys.exit(1)
            n += 1
        os.chmod(tmp, p.stat().st_mode & 0o777)
        os.replace(tmp, p)
        print(f"  REWRITTEN       {rel}  ({n} vocabularies, transforms identical; "
              f"sha {old_sha[:12]} -> {sha(p)[:12]})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
