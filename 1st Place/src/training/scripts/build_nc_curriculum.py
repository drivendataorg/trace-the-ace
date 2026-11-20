"""Build artifacts/nc_curriculum.csv from the National Curriculum in England (mathematics).

One-off script that built the shipped curriculum table (data/external/nc_curriculum.csv, copied
into the workspace by src/make_workspace.py). It is not a reproduce.py step: it scrapes the live
gov.uk page, which may have changed since, so a re-run can differ from the shipped file.

## Data source

UK National Curriculum Mathematics Programmes of Study (statutory, England):
  https://www.gov.uk/government/publications/national-curriculum-in-england-mathematics-
  programmes-of-study/national-curriculum-in-england-mathematics-programmes-of-study

Published by the Department for Education; Crown copyright / Open Government Licence.

## How nc_curriculum.csv is built

This script scrapes the live DfE page using Python's stdlib `urllib` + `html.parser`
(no BeautifulSoup or other third-party dependencies). The HTML is well-structured:
year-group sections are h2 headings, strand subsections are h3/h4 headings, and
statutory statements are <li> elements. Notes-and-guidance (non-statutory) h4 sections
are identified by their heading ID prefix and skipped.

Nested list handling: many NC statements use nested <ul> (e.g. "compare, describe and
solve practical problems for: <ul><li>lengths and heights</li> ...")). Each child <li>
is emitted as a full statement with the parent li text prepended:
    "compare, describe and solve practical problems for: lengths and heights [...]"
This keeps the predicate context that TF-IDF matching needs.

Coverage: Y1 to Y6 (KS1/KS2, exact year labels), KS3 -> year_group=8 (Y7-Y9 midpoint),
KS4 -> year_group=10 (Y10-Y11 midpoint).
Strands: Number, Algebra, Geometry (Measurement folded in), Statistics, Ratio.
Each row: year_group, ks_level, topic_strand, sub_strand, statement.

## LO matching (lo_nc_match.csv)

TF-IDF (unigrams and bigrams, sublinear_tf) fitted on the NC statements plus all 398 training
LO texts -> cosine similarity matrix (398 LOs x N NC statements).

For each LO:
  - `cosine_sim`: max similarity to any NC statement
  - `year_group_soft`: similarity-weighted average year_group of the top 5 matches (an
    objective can match several years with similar scores: "Telling the time" spans Y1-Y3)
  - `ks_level`: ks_level of the single best-match statement
  - `strand_match`: topic_strand of the best-match statement

At inference the NC features are recomputed from the bundled `assets/nc_curriculum.csv` and
the test LO texts (`_nc_match_live` in features.py). No precomputed per-LO file is needed, so
objectives unseen in training are handled.

Out:
  artifacts/nc_curriculum.csv   scraped NC statements (bundled into assets/)
  artifacts/lo_nc_match.csv     the 398 training LOs: best match, year_group_soft, cosine_sim

Usage (from build/workspace/, needs internet access):
    PYTHONPATH=src python scripts/build_nc_curriculum.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from magnificat import config, data

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

_NC_URL = (
    "https://www.gov.uk/government/publications/"
    "national-curriculum-in-england-mathematics-programmes-of-study/"
    "national-curriculum-in-england-mathematics-programmes-of-study"
)


def _map_strand_y1y6(text: str) -> tuple[str, str] | tuple[None, None]:
    """Map a Y1-Y6 h3 heading text to (topic_strand, sub_strand)."""
    t = text.lower().strip()
    if t.startswith("number"):
        parts = text.split(" - ", 1)
        sub = parts[1].split("(")[0].strip() if len(parts) > 1 else "Number"
        return "Number", sub
    if "measurement" in t:
        return "Geometry", "Measurement"
    if t.startswith("geometry"):
        parts = text.split(" - ", 1)
        sub = parts[1].strip() if len(parts) > 1 else "Geometry"
        return "Geometry", sub
    if "ratio" in t or "proportion" in t:
        return "Ratio", "Ratio and Proportion"
    if t.startswith("algebra"):
        return "Algebra", "Algebra"
    if "statistics" in t:
        return "Statistics", "Statistics"
    return None, None


def _map_strand_ks34(text: str) -> tuple[str, str] | tuple[None, None]:
    """Map a KS3/KS4 strand heading text to (topic_strand, sub_strand)."""
    t = text.lower().strip()
    if t.startswith("number"):
        return "Number", "Number"
    if t.startswith("algebra"):
        return "Algebra", "Algebra"
    if "ratio" in t or "proportion" in t or "rates" in t:
        return "Ratio", "Ratio, Proportion and Rates of Change"
    if "geometry" in t or "measure" in t:
        return "Geometry", "Geometry and Measures"
    if "probability" in t:
        return "Statistics", "Probability"
    if "statistics" in t:
        return "Statistics", "Statistics"
    return None, None


class _NCParser(HTMLParser):
    """State-machine HTML parser that extracts statutory NC statements.

    Handles nested bullet lists by prepending the parent li text to each child:

        parent: "compare, describe and solve practical problems for"
        child:  "lengths and heights [for example, long/short, ...]"
        emits:  "compare, describe and solve practical problems for: lengths and heights [...]"

    For non-nested bullets the li text is emitted as-is.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict] = []

        # current heading being read
        self._htag: str | None = None
        self._hid: str = ""
        self._hbuf: list[str] = []

        # Nested-list context stack.  Each entry is a dict with keys:
        #   "ctx"     parent li text (before the nested <ul>); '' for top level
        #   "nested"  True if this frame was pushed from inside a <li>
        self._ul_stack: list[dict] = [{"ctx": "", "nested": False}]
        self._in_li = False
        self._libuf: list[str] = []

        # curriculum state
        self._year = 0
        self._ks = 0
        self._strand: str | None = None
        self._sub: str | None = None
        self._in_notes = False
        self._in_subject_content = False

    # ── HTMLParser callbacks ─────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_d = dict(attrs)
        hid = attrs_d.get("id", "")

        if tag in ("h2", "h3", "h4"):
            self._htag = tag
            self._hid = hid
            self._hbuf = []
            self._in_li = False

        elif tag == "ul":
            if self._in_li:
                # Nested <ul> inside a <li>: save parent li text as context.
                parent_ctx = " ".join("".join(self._libuf).split()).rstrip(":")
                self._ul_stack.append({"ctx": parent_ctx, "nested": True})
                self._libuf = []
            else:
                # Top-level <ul>: push a non-contextual sentinel.
                self._ul_stack.append({"ctx": "", "nested": False})

        elif tag == "li":
            self._in_li = True
            self._libuf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3", "h4") and self._htag == tag:
            text = "".join(self._hbuf).strip()
            self._process_heading(tag, self._hid, text)
            self._htag = None
            self._hbuf = []

        elif tag == "ul":
            if len(self._ul_stack) > 1:
                self._ul_stack.pop()

        elif tag == "li" and self._in_li:
            child = " ".join("".join(self._libuf).split())
            top = self._ul_stack[-1] if self._ul_stack else {"ctx": "", "nested": False}
            if top["nested"]:
                # Prepend parent context to each child individually.
                ctx = top["ctx"]
                stmt = f"{ctx}: {child}" if ctx else child
                self._emit(stmt.strip())
            else:
                # Top-level li: emit as-is.
                self._emit(child)
            self._in_li = False
            self._libuf = []

    def handle_data(self, data: str) -> None:
        if self._htag:
            self._hbuf.append(data)
        elif self._in_li:
            self._libuf.append(data)

    def _emit(self, stmt: str) -> None:
        stmt = stmt.strip()
        if stmt and not self._in_notes and self._year > 0 and self._strand:
            self.records.append({
                "year_group":   self._year,
                "ks_level":     self._ks,
                "topic_strand": self._strand,
                "sub_strand":   self._sub or "",
                "statement":    stmt,
            })

    # ── State machine ────────────────────────────────────────────────────────

    def _process_heading(self, tag: str, hid: str, text: str) -> None:
        if tag == "h2":
            self._handle_h2(hid)
        elif tag == "h3":
            self._handle_h3(hid, text)
        elif tag == "h4":
            self._handle_h4(hid, text)

    def _handle_h2(self, hid: str) -> None:
        self._strand = None
        self._sub = None
        self._in_notes = False
        self._in_subject_content = False

        m = re.match(r"year-(\d+)-programme-of-study", hid)
        if m:
            y = int(m.group(1))
            self._year = y
            self._ks = 1 if y <= 2 else 2
        elif hid == "key-stage-3":
            self._year = 8
            self._ks = 3
        elif hid == "key-stage-4":
            self._year = 10
            self._ks = 4
        else:
            self._year = 0  # key-stage intro sections: ignored

    def _handle_h3(self, hid: str, text: str) -> None:
        if "notes-and-guidance" in hid:
            self._in_notes = True
            return
        self._in_notes = False

        if self._ks in (3, 4):
            if hid in ("subject-content", "subject-content-1"):
                self._in_subject_content = True
                return
            if self._ks == 4 and self._in_subject_content:
                strand, sub = _map_strand_ks34(text)
                self._strand = strand
                self._sub = sub
            # else: KS3 uses h4 for strands; its h3 is intro / working mathematically: skip
        else:
            # Y1-Y6
            strand, sub = _map_strand_y1y6(text)
            self._strand = strand
            self._sub = sub

    def _handle_h4(self, hid: str, text: str) -> None:
        if "notes-and-guidance" in hid:
            self._in_notes = True
            return
        self._in_notes = False

        if self._ks == 3 and self._in_subject_content:
            strand, sub = _map_strand_ks34(text)
            self._strand = strand
            self._sub = sub


def scrape_nc(url: str = _NC_URL) -> pd.DataFrame:
    """Fetch and parse the DfE National Curriculum mathematics page.

    Args:
        url: page to fetch (default: the gov.uk programmes of study).

    Returns:
        DataFrame of statutory statements (year_group, ks_level, topic_strand, sub_strand,
        statement), with example clauses stripped and duplicates removed.
    """
    print(f"Fetching {url} …")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")

    # Narrow to the govspeak article body (avoids nav / footer noise)
    m = re.search(r'<div class="govspeak">(.*?)</div>\s*</div>\s*</div>\s*</div>',
                  html, re.DOTALL)
    body = m.group(1) if m else html

    parser = _NCParser()
    parser.feed(body)
    df = pd.DataFrame(parser.records)
    print(f"Parsed {len(df)} statutory statements.")

    df = _clean_statements(df)
    print(f"After cleaning: {len(df)} statements.")
    return df


# Strips "[for example, ...]" clauses: ~5-10 words per statement that dominate TF-IDF
# without adding meaning for matching.
# e.g. "lengths and heights [for example, long/short, longer/shorter, ...]"
#   -> "lengths and heights"
_EXAMPLE_RE = re.compile(r"\s*\[for example[^\]]*\]", re.IGNORECASE)


def _clean_statements(df: pd.DataFrame) -> pd.DataFrame:
    """Strip "[for example, ...]" noise clauses and deduplicate."""
    df = df.copy()
    df["statement"] = (
        df["statement"]
        .str.replace(_EXAMPLE_RE, "", regex=True)
        .str.strip()
    )
    df = df[df["statement"].str.len() > 5]
    df = df.drop_duplicates(subset=["year_group", "topic_strand", "statement"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main: build nc_curriculum.csv and lo_nc_match.csv
# ---------------------------------------------------------------------------

def main() -> None:
    """Scrape the curriculum and match every training objective to it.

    Returns:
        None. Writes artifacts/nc_curriculum.csv and artifacts/lo_nc_match.csv and prints
        the strand, year and key-stage distributions and the low-similarity matches.
    """
    df = scrape_nc()
    out = config.ARTIFACTS_DIR / "nc_curriculum.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} NC statements → {out}")

    # ── Match each LO to closest NC statement via TF-IDF cosine similarity ──
    feats = data.load_features()
    los = (feats.drop_duplicates("learning_objective_id")
               [["learning_objective_id", "learning_objective"]]
               .reset_index(drop=True))
    print(f"Matching {len(los)} LOs to {len(df)} NC statements …")

    corpus = list(df["statement"]) + list(los["learning_objective"])

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          strip_accents="unicode", min_df=1)
    vec.fit(corpus)
    nc_vecs = vec.transform(list(df["statement"]))
    lo_vecs = vec.transform(list(los["learning_objective"]))

    sims = cosine_similarity(lo_vecs, nc_vecs)  # (n_lo, n_nc)
    best_score = sims.max(axis=1)

    # Soft top-k: similarity-weighted average year_group over top-5 NC matches
    TOP_K = 5
    top_k_idx   = np.argsort(sims, axis=1)[:, -TOP_K:]
    top_k_sims  = sims[np.arange(len(los))[:, None], top_k_idx]
    top_k_years = np.array(df["year_group"].values)[top_k_idx]
    sim_sum     = top_k_sims.sum(axis=1, keepdims=True).clip(min=1e-9)
    year_soft   = (top_k_sims * top_k_years).sum(axis=1) / sim_sum.squeeze()

    best_idx = sims.argmax(axis=1)

    result_rows = []
    for i, (_, lo_row) in enumerate(los.iterrows()):
        nc_row = df.iloc[best_idx[i]]
        result_rows.append({
            "lo_id":            lo_row["learning_objective_id"],
            "lo_text":          lo_row["learning_objective"],
            "nc_statement":     nc_row["statement"],
            "year_group":       int(nc_row["year_group"]),
            "year_group_soft":  round(float(year_soft[i]), 3),
            "ks_level":         int(nc_row["ks_level"]),
            "topic_strand":     nc_row["topic_strand"],
            "sub_strand":       nc_row["sub_strand"],
            "cosine_sim":       round(float(best_score[i]), 4),
        })

    out_df = pd.DataFrame(result_rows).sort_values("lo_id")
    match_out = config.ARTIFACTS_DIR / "lo_nc_match.csv"
    out_df.to_csv(match_out, index=False)
    print(f"Saved {len(out_df)} LO matches → {match_out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\nStrand distribution (NC-matched):")
    print(out_df["topic_strand"].value_counts().to_string())
    print("\nYear group distribution:")
    print(out_df["year_group"].value_counts().sort_index().to_string())
    print("\nKS distribution:")
    print(out_df["ks_level"].value_counts().sort_index().to_string())
    print(f"\nMedian cosine similarity: {out_df['cosine_sim'].median():.3f}")
    print(f"Low-confidence matches (sim < 0.10): {(out_df['cosine_sim'] < 0.10).sum()}")

    low = out_df[out_df["cosine_sim"] < 0.10].sort_values("cosine_sim")
    if len(low):
        print("\nLow-confidence LOs (<0.10 sim):")
        for _, r in low.iterrows():
            print(f"  sim={r['cosine_sim']:.3f}  {r['lo_text']!r}")
            print(f"    → matched: {r['nc_statement']!r}  [{r['topic_strand']} Y{r['year_group']}]")


if __name__ == "__main__":
    main()
