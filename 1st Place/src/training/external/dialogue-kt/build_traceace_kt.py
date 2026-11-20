#!/usr/bin/env python
"""Convert the Trace the Ace training data to the LLMKT annotated-DataFrame format.

Each (session, learning objective) response becomes one "dialogue" sample:
  - dialogue   : the session transcript as [{turn, teacher, student}, ...] plus a synthetic,
                 empty terminal turn whose prediction is the label.
  - annotation : every real turn is `na` (correct=None); only the terminal turn carries
                 kcs=[LO text] and correct=is_correct. LLMKT therefore trains and evaluates on
                 end-of-session correctness for that objective (the analogue of the dialogue-kt
                 paper's "Final Turn" metric), with the full transcript available as context
                 and no label leak.
  - meta_data  : {expected_result, response_id, session_id, lo_id}; expected_result drives
                 apply_annotations' terminal-turn correctness override.

Role mapping: tutor -> teacher, student -> student, background -> dropped (noise/[unclear]).
Output: data/annotated/traceace.pkl by default (--out); not shipped, because it holds
competition text. Step 01a of the pipeline runs
`build_traceace_kt.py --select full --max_turns 0 --out data/annotated/traceace_full.pkl`.

The options --turn_kcs and --kc_source atc, nc or session (session also needs --kc_turns)
import modules or read files that are not in this repository, so they do not run here.
"""
import argparse, glob, os, random, sys
import pandas as pd

# repo modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialogue_kt.data_loading import process_dialogue

MAGNIFICAT = os.environ.get("MAGNIFICAT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # default: the training/ root
DATA = f"{MAGNIFICAT}/data"

# Permissive (MIT), offline-bundleable encoder for LO->turn retrieval (--select lo_relevant).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def find_one(pattern):
    """Return the first path (in sorted order) that matches a glob pattern.

    Args:
        pattern: glob pattern, e.g. ``data/train_features*.csv``.

    Returns:
        The matching path.

    Raises:
        FileNotFoundError: if nothing matches.
    """
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def turn_text(t: dict) -> str:
    """Join a turn's teacher and student text into one string for embedding.

    Args:
        t: a turn dict with optional ``teacher`` and ``student`` keys.

    Returns:
        The joined text, stripped (empty if the turn has no text).
    """
    parts = []
    if t.get("teacher"):
        parts.append(str(t["teacher"]))
    if t.get("student"):
        parts.append(str(t["student"]))
    return " ".join(parts).strip()


class LORetriever:
    """Rank a session's turns by cosine similarity to a learning objective.

    Each session's turns are embedded once (sessions are shared across responses) and cached,
    then re-ranked per objective. sentence-transformers is imported lazily, so the default
    (`--select last` / `full`) path does not need it.

    Args:
        model_name: sentence-transformers model id (default EMBED_MODEL).
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy
        self.model = SentenceTransformer(model_name)
        self._turn_cache = {}   # session_id -> (turn_embeddings [N,d], normalized)
        self._lo_cache = {}     # lo text     -> embedding [d]

    def _embed(self, texts):
        # normalized embeddings -> cosine == dot product
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=64)

    def turn_embeddings(self, session_id, turns):
        """Return the cached [n_turns, d] normalised embeddings of a session's turns."""
        cached = self._turn_cache.get(session_id)
        if cached is None:
            cached = self._embed([turn_text(t) or " " for t in turns])
            self._turn_cache[session_id] = cached
        return cached

    def lo_embedding(self, lo_text):
        """Return the cached [d] normalised embedding of an objective text."""
        emb = self._lo_cache.get(lo_text)
        if emb is None:
            emb = self._embed([lo_text])[0]
            self._lo_cache[lo_text] = emb
        return emb

    def select(self, session_id, turns, lo_text, top_k, end_window):
        """Keep the turns most similar to the objective plus the last turns of the session.

        Args:
            session_id: session id (cache key for the turn embeddings).
            turns: the session's turn list.
            lo_text: learning-objective text.
            top_k: number of most similar turns to keep (0 = none).
            end_window: number of trailing turns always kept.

        Returns:
            The selected turns in chronological order, without duplicates.
        """
        import numpy as np
        n = len(turns)
        keep = set(range(max(0, n - end_window), n))   # end-of-session window always kept
        if top_k > 0 and n > 0:
            te = self.turn_embeddings(session_id, turns)
            scores = te @ self.lo_embedding(lo_text)
            keep.update(int(i) for i in np.argsort(-scores)[:top_k])
        return [turns[i] for i in sorted(keep)]

def build_session_dialogue(tx: pd.DataFrame):
    """Convert one session's transcript rows to an LLMKT turn list.

    Args:
        tx: transcript DataFrame (utterance_id, role, content); `background` rows and empty
            utterances are dropped.

    Returns:
        The turn list from ``dialogue_kt.data_loading.process_dialogue``, or None if the
        session has no tutor or student utterance.
    """
    tx = tx.sort_values("utterance_id")
    utts = []
    for _, u in tx.iterrows():
        role = str(u["role"]).strip().lower()
        content = "" if pd.isna(u["content"]) else str(u["content"]).strip()
        if not content:
            continue
        if role == "tutor":
            utts.append({"role": "teacher", "content": content})
        elif role == "student":
            utts.append({"role": "student", "content": content})
        # background -> dropped
    if not utts:
        return None
    return process_dialogue(utts)

def main():
    """Build the LLMKT table from the training CSVs and transcripts (options: see --help).

    Returns:
        None. Writes the pickle named by --out and prints counts and the label balance.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of responses (for a quick build); keeps LO diversity")
    ap.add_argument("--select", choices=["last", "full", "lo_relevant"], default="last",
                    help="turn selection: last N (end window, the default), full transcript, "
                         "or top-K LO-relevant + end window")
    ap.add_argument("--max_turns", type=int, default=40,
                    help="--select last: keep only the last N turns (0 = keep all)")
    ap.add_argument("--top_k", type=int, default=20,
                    help="--select lo_relevant: # of most-LO-similar turns to keep")
    ap.add_argument("--end_window", type=int, default=20,
                    help="--select lo_relevant: # of trailing turns to always keep")
    ap.add_argument("--out", type=str, default="data/annotated/traceace.pkl")
    ap.add_argument("--turn_labels", type=str, default=None,
                    help="csv (session_id,turn,label) of per-turn correctness pseudo-labels. "
                         "Sampled true/false turns become labeled KT waypoints (kc = the "
                         "dialogue's LO); 'na' ignored. Requires --select full (annotation turn "
                         "indices are global).")
    ap.add_argument("--max_pseudo_per_dialogue", type=int, default=4,
                    help="--turn_labels: cap of pseudo-labeled turns sampled per dialogue")
    ap.add_argument("--turn_kcs", type=str, default=None,
                    help="--turn_labels: give each pseudo-labelled waypoint its own KCs from a "
                         "per-turn KC csv (session_id,turn,kc_ids) instead of the session LO; a "
                         "waypoint is kept only if it has both a true/false label and >=1 KC.")
    ap.add_argument("--kc_turns", type=str, default=None,
                    help="--kc_source session: per-turn KC csv (session_id,turn,kc_ids); the "
                         "terminal turn's KCs are taken from the session's own annotated turns.")
    ap.add_argument("--kc_window", type=int, default=12,
                    help="--kc_source session: how many trailing turns to draw KCs from.")
    ap.add_argument("--kc_max", type=int, default=4,
                    help="--kc_source session: keep the K most frequent KCs in the window.")
    ap.add_argument("--kc_source", choices=["lo", "atc", "nc", "session"], default="lo",
                    help="what fills the `kcs` slot that LLMKT queries. 'lo' (default, used by "
                         "the shipped model): the raw learning-objective string, exactly one KC "
                         "per sample, which makes --agg mean-ar a no-op. 'atc'/'nc': the LO "
                         "decomposed into curriculum standards; 'session': KCs from --kc_turns. "
                         "The last three need files or modules that are not in this repository.")
    ap.add_argument("--kc_assign", choices=["funnel", "retriever"], default="funnel",
                    help="--kc_source atc|nc: use the stored LLM funnel assignment, or reproduce "
                         "it with an encoder-based retriever (needs a module that is not in this "
                         "repository)")
    ap.add_argument("--kc_top_k", type=int, default=3,
                    help="--kc_assign retriever: KCs per LO")
    args = ap.parse_args()
    if args.turn_labels and args.select != "full":
        ap.error("--turn_labels requires --select full")

    # session_id -> [kc text, ...], from the session's own annotated turns
    sess_kc = {}
    if args.kc_source == "session":
        if not args.kc_turns:
            ap.error("--kc_source session requires --kc_turns")
        from collections import Counter
        from annotate_traceace_kc_vllm import load_taxonomy
        tax = load_taxonomy(os.environ.get("ANNO_TAXONOMY", "atc"))
        kt = pd.read_csv(args.kc_turns)
        kt = kt[kt.kc_ids.notna()]
        for sid, g in kt.groupby("session_id"):
            tail = g.nlargest(args.kc_window, "turn")          # end-of-session state
            ids = [x for row in tail.kc_ids for x in str(row).split("|") if x]
            top = [k for k, _ in Counter(ids).most_common(args.kc_max)]
            texts = [tax["standards"][k]["description"] for k in top if k in tax["standards"]]
            if texts:
                sess_kc[sid] = texts
        sizes = [len(v) for v in sess_kc.values()]
        print(f"KC source session: {len(sess_kc)} sessions mapped, "
              f"mean {sum(sizes)/max(len(sizes),1):.2f} KCs "
              f"(window {args.kc_window} turns, cap {args.kc_max}); "
              f"sessions without KCs fall back to the LO string")

    # lo_id -> [kc text, ...]; empty dict means "keep the default LO-as-KC behaviour"
    kc_map = {}
    if args.kc_source in ("atc", "nc"):
        from kc_assign import load_lo_kcs
        lo_kcs = load_lo_kcs(args.kc_source)
        if args.kc_assign == "retriever":
            from kc_assign import TaxonomyRetriever
            from annotate_traceace_kc_vllm import load_taxonomy
            tax = load_taxonomy(args.kc_source)
            r = TaxonomyRetriever(tax)
            kc_map = {lo: [tax["standards"][k]["description"]
                           for k in r.assign(v["lo_text"], k=args.kc_top_k)]
                      for lo, v in lo_kcs.items()}
        else:
            kc_map = {lo: list(v["kc_texts"]) for lo, v in lo_kcs.items()}
        kc_map = {lo: kcs for lo, kcs in kc_map.items() if kcs}
        sizes = [len(v) for v in kc_map.values()]
        print(f"KC source {args.kc_source}/{args.kc_assign}: {len(kc_map)} LOs mapped, "
              f"mean {sum(sizes)/max(len(sizes),1):.2f} KCs/LO "
              f"(LOs without a mapping fall back to the LO string)")

    turn_kc = {}
    if args.turn_kcs:
        from annotate_traceace_kc_vllm import load_taxonomy
        _tax = load_taxonomy(os.environ.get("ANNO_TAXONOMY", "atc"))
        _tk = pd.read_csv(args.turn_kcs)
        _tk = _tk[_tk.kc_ids.notna()]
        for _sid, _turn, _ids in zip(_tk.session_id, _tk.turn, _tk.kc_ids):
            _txt = [_tax["standards"][k]["description"]
                    for k in str(_ids).split("|") if k in _tax["standards"]]
            if _txt:
                turn_kc[(_sid, int(_turn))] = _txt
        print(f"per-turn KCs: {len(turn_kc):,} (session, turn) pairs carry >=1 KC")

    pseudo = {}
    if args.turn_labels:
        tl = pd.read_csv(args.turn_labels)
        tl = tl[tl.label.isin(["true", "false"])]
        pseudo = {sid: list(zip(g.turn, g.label == "true")) for sid, g in tl.groupby("session_id")}
        print(f"pseudo-labels: {sum(len(v) for v in pseudo.values())} true/false turns "
              f"over {len(pseudo)} sessions (cap {args.max_pseudo_per_dialogue}/dialogue)")

    feats = pd.read_csv(find_one(f"{DATA}/train_features*.csv"))
    labels = pd.read_csv(find_one(f"{DATA}/train_labels*.csv"))
    df = feats.merge(labels, on="response_id", how="inner")
    print(f"merged responses: {len(df)}  unique LO: {df.learning_objective_id.nunique()}  sessions: {df.session_id.nunique()}")

    if args.limit:
        # sample responses but keep all LOs represented: 1 per LO first, then fill
        first = df.groupby("learning_objective_id", group_keys=False).head(1)
        rest = df.drop(first.index)
        take = max(0, args.limit - len(first))
        df = pd.concat([first, rest.sample(n=min(take, len(rest)), random_state=221)]).sort_index()
        print(f"limited to {len(df)} responses across {df.learning_objective_id.nunique()} LOs")

    # cache dialogues per session (sessions are shared across responses)
    dia_cache = {}
    rows, skipped, n_way = [], 0, 0
    sessions = df.session_id.unique()
    for i, sid in enumerate(sessions):
        if i % 2000 == 0:
            print(f"  transcripts {i}/{len(sessions)}", flush=True)
        path = f"{DATA}/train_transcripts/{sid}.csv"
        if not os.path.exists(path):
            dia_cache[sid] = None
            continue
        dia_cache[sid] = build_session_dialogue(pd.read_csv(path))

    retriever = LORetriever() if args.select == "lo_relevant" else None
    for _, r in df.iterrows():
        base = dia_cache.get(r["session_id"])
        if not base:
            skipped += 1
            continue
        real = [dict(t) for t in base]                # copy (apply_annotations mutates)
        if args.select == "lo_relevant":
            real = retriever.select(r["session_id"], real, str(r["learning_objective"]).strip(),
                                    args.top_k, args.end_window)
        elif args.select == "last" and args.max_turns and len(real) > args.max_turns:
            real = real[-args.max_turns:]             # end-of-session window
        # --select full: keep every turn
        for i, t in enumerate(real, start=1):         # renumber 1..M (contiguous)
            t["turn"] = i
        term = len(real) + 1
        dialogue = real + [{"turn": term, "teacher": "", "student": ""}]
        is_correct = bool(round(float(r["is_correct"])))
        # The KC set LLMKT queries at the terminal turn. Default: the LO string itself
        # (K=1). With --kc_source atc|nc: the LO decomposed into curriculum standards,
        # so `--agg mean-ar` averages K real per-KC mastery reads (paper Eq. 1).
        lo_text = str(r["learning_objective"]).strip()
        # session-derived KCs take precedence; then the LO decomposition; then the raw LO
        term_kcs = (sess_kc.get(r["session_id"])
                    or kc_map.get(str(r["learning_objective_id"]))
                    or [lo_text])
        anno = {f"turn {t['turn']}": {"correct": None, "kcs": []} for t in real}
        if pseudo:
            # sample capped pseudo waypoints for this dialogue (deterministic per response)
            cand = [(t, c) for t, c in pseudo.get(r["session_id"], []) if 1 <= t <= len(real)]
            rng = random.Random(f"{r['response_id']}-221")
            if turn_kc:   # the paper's mutual gate: keep a waypoint only if it also has KCs
                cand = [(t, c) for t, c in cand if (r["session_id"], t) in turn_kc]
            for t, c in (rng.sample(cand, args.max_pseudo_per_dialogue)
                         if len(cand) > args.max_pseudo_per_dialogue else cand):
                kcs_t = turn_kc.get((r["session_id"], t), list(term_kcs))
                anno[f"turn {t}"] = {"correct": bool(c), "kcs": list(kcs_t)}
                n_way += 1
        anno[f"turn {term}"] = {"correct": is_correct, "kcs": list(term_kcs)}
        rows.append({
            "index": r["response_id"],
            "dialogue": dialogue,
            "annotation": anno,
            "meta_data": {
                "expected_result": "Answer Accepted" if is_correct else "Answer Not Accepted",
                "response_id": r["response_id"],
                "session_id": r["session_id"],
                "lo_id": r["learning_objective_id"],
            },
            "lo_id": r["learning_objective_id"],
        })

    out = pd.DataFrame(rows)
    # pandas >= 3 gives text columns ("index", "lo_id") and the column labels a string dtype, and
    # the stage-1 LoRA env (pandas 1.5.3) cannot unpickle that (TypeError: issubclass() arg 1 must
    # be a class). Store both as plain object, as pandas 1.5.3 does; there this is a no-op, so the
    # pickle written under KT_PYTHON is unchanged by it. Casting only one of the two is not
    # enough.
    txt = [c for c in out.columns if pd.api.types.is_string_dtype(out[c]) and out[c].dtype != object]
    if txt:
        out = out.astype({c: object for c in txt})
    if out.columns.dtype != object:
        out.columns = out.columns.astype(object)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_pickle(args.out)
    print(f"wrote {len(out)} samples -> {args.out}  (skipped {skipped} missing/empty transcripts)")
    def _term_label(row):
        term = row["dialogue"][-1]["turn"]
        return row["annotation"]["turn " + str(term)]["correct"]
    if args.turn_labels:
        print(f"waypoints: {n_way:,} over {len(rows):,} samples "
              f"(mean {n_way/max(len(rows),1):.2f}/dialogue"
              f"{', own-KC' if turn_kc else ', LO-as-KC'})")
    print(f"label balance: {out.apply(_term_label, axis=1).mean():.3f} correct")

if __name__ == "__main__":
    main()
