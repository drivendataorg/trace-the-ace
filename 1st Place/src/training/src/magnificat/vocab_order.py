"""Store fitted TF-IDF vocabularies in sorted order before a model is saved.

scikit-learn builds ``vocabulary_`` while it reads the training text, so the dict keeps its
terms in first-occurrence order: read in order, the word bigrams and character n-grams of a
fitted vectorizer replay its training transcripts. ``sort_vocabularies`` rebuilds every fitted
``vocabulary_`` inside an object as the same term -> column mapping, with the keys in sorted
order. ``transform`` only looks terms up, so the model's outputs are unchanged bit for bit, but
the saved file no longer preserves the order of the training text.

Every training script that saves a vectorizer saves it with ``dump_sorted``.
"""
from __future__ import annotations


def sort_vocabularies(obj, _seen=None) -> int:
    """Sort, in place, the ``vocabulary_`` of every fitted vectorizer reachable from ``obj``.

    Walks dicts, lists, tuples and object attributes recursively, visiting each object once.

    Args:
        obj: a fitted model, pipeline or container of them.
        _seen: ids of objects already visited (internal, for the recursion).

    Returns:
        The number of vocabularies sorted.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    seen = set() if _seen is None else _seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, CountVectorizer):
        voc = getattr(obj, "vocabulary_", None)
        if voc:
            obj.vocabulary_ = dict(sorted(voc.items()))
            return 1
        return 0
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple)):
        children = obj
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        children = vars(obj).values()
    else:
        return 0
    return sum(sort_vocabularies(c, seen) for c in children)


def resave_sorted(path, compress=0) -> None:
    """Load ``path``, sort its vocabularies and save it again, in a fresh Python process.

    The output's bytes depend only on the input file's bytes: pickle shares a string between two
    places only if it is the same object in memory, and which strings a process already holds
    changes that. A fresh process for every file makes the result reproducible, so a retrained
    model whose unsorted save matches the original byte for byte also matches the shipped,
    sorted file byte for byte.

    Args:
        path: joblib file to rewrite in place.
        compress: joblib compression level.

    Returns:
        None. Overwrites ``path``.
    """
    import json
    import subprocess
    import sys

    code = ("import json, sys; sys.path[:0] = json.loads(sys.argv[3]); import joblib; "
            "from magnificat.vocab_order import sort_vocabularies; "
            "o = joblib.load(sys.argv[1]); sort_vocabularies(o); "
            "joblib.dump(o, sys.argv[1], compress=int(sys.argv[2]))")
    here = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    subprocess.run([sys.executable, "-B", "-c", code, str(path), str(compress),
                    json.dumps([here] + [p for p in sys.path if p])], check=True)


def dump_sorted(obj, path, compress=0) -> None:
    """Save ``obj`` with ``joblib.dump``, then rewrite it with sorted vocabularies.

    Args:
        obj: the object to save.
        path: output file.
        compress: joblib compression level.

    Returns:
        None. Writes ``path`` (see ``resave_sorted``).
    """
    import joblib

    joblib.dump(obj, path, compress=compress)
    resave_sorted(path, compress)
