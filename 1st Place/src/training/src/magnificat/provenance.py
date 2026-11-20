"""Licence provenance stamped into trained artifacts, and the guard that reads it.

Prize eligibility requires commercially licensable data, so non-commercial (NC) corpora must
not reach anything that ships. A filename convention cannot enforce that on its own: staging
copies each model to a fixed name inside `assets/` (e.g. `baseline.joblib`), so the source
filename, and any marker in it, is lost.

So the provenance travels inside the object. `stamp()` attaches it at `joblib.dump` time;
`assert_shippable()` reads it and exits on a violation.

It also covers a less obvious case: a loknn neighbourhood table
(`artifacts/loknn_table.joblib`) is a statistic of the training labels
(`lo_knn.fit(lo[tr], y[tr], ...)`). If NC rows are in that fit, NC-derived labels would ship
inside the table even when the booster itself does not, so the table is stamped too.

The licence classes are coarse on purpose: the only question asked is "may this ship?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Corpora that may never touch a shipped artifact. QATD's dataset card says
#: `license: cc-by-nc-4.0`; Bridge is CC BY-NC 4.0 on the K-12 AI Infrastructure Platform and
#: Hugging Face (THIRD_PARTY_NOTICES.md).
NON_COMMERCIAL = {"qatd", "bridge", "comta", "talkmoves"}

#: Permissive or share-alike; shippable. `mrbench` and `mathdial` are CC BY-SA.
#:
#: CC BY-SA is commercially licensable, so it meets the licence rule ("no NC / CC-NC /
#: CC-BY-NC"), and BY-SA corpora may be used, including in shipped artifacts. BY-SA imposes
#: attribution and share-alike on derivatives, which are obligations to meet, not a bar to use;
#: they are met by declaring and documenting the corpora with their licences (the challenge
#: rules: "Finalists must declare and document all external resources"), not by a packer gate.
#: So there is no SHARE_ALIKE set here: this module answers one question, "may this ship?", and
#: for BY-SA the answer is yes.
#:
#: About 20% of MRBench is derived from Bridge by content (107 of 504 dialogues; see
#: data/external/mrbench/NOTICE.md); `mrbench` is classed shippable by its entry in this set.
SHIPPABLE = {"tta", "mrbench", "mathdial", "book2dial", "xes3g5m",
             # `ttaprefix*`: Trace the Ace transcripts cut before a graded turn + our own 122B
             # per-turn labels; Trace the Ace data under its licence. TERM/SHUF are label swaps.
             "ttaprefix", "ttaprefix_TERM", "ttaprefix_SHUF"}


@dataclass
class Provenance:
    """Which corpora touched a fit, and whether the result may ship.

    Attributes:
        train_corpora: corpora whose rows entered the model fit.
        loknn_corpora: corpora whose labels entered the loknn neighbourhood table.
        note: free-text note.
        extra: any further metadata.
    """
    train_corpora: tuple = ()
    #: corpora whose labels entered the loknn neighbourhood table specifically
    loknn_corpora: tuple = ()
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def nc_corpora(self) -> tuple:
        """Every non-commercial corpus that touched either the booster fit or the loknn table."""
        touched = set(self.train_corpora) | set(self.loknn_corpora)
        return tuple(sorted(touched & NON_COMMERCIAL))

    @property
    def never_ship(self) -> bool:
        return bool(self.nc_corpora)

    # One-line summary: "train_corpora=[...] loknn=[...] -> NEVER_SHIP|shippable".
    def describe(self) -> str:
        c = ", ".join(self.train_corpora) or "(unstamped)"
        s = "NEVER_SHIP" if self.never_ship else "shippable"
        return f"train_corpora=[{c}] loknn=[{', '.join(self.loknn_corpora)}] -> {s}"


def stamp(obj, train_corpora, loknn_corpora=(), note=""):
    """Attach provenance to a fitted object, in place, and return it.

    Args:
        obj: the fitted model or table (gets a ``_provenance`` attribute).
        train_corpora: names of the corpora whose rows entered the fit.
        loknn_corpora: names of the corpora whose labels entered a loknn table.
        note: free-text note.

    Returns:
        ``obj``, with ``obj._provenance`` set to a :class:`Provenance`.

    Raises:
        ValueError: a corpus name is in neither ``NON_COMMERCIAL`` nor ``SHIPPABLE`` (an
            unclassified corpus must not default to shippable).
    """
    train_corpora = tuple(sorted(set(train_corpora)))
    loknn_corpora = tuple(sorted(set(loknn_corpora)))
    unknown = sorted((set(train_corpora) | set(loknn_corpora))
                     - NON_COMMERCIAL - SHIPPABLE)
    if unknown:
        raise ValueError(
            f"unclassified corpora {unknown}: add them to magnificat.provenance."
            "NON_COMMERCIAL or SHIPPABLE. Refusing to default to shippable.")
    obj._provenance = Provenance(train_corpora=train_corpora,
                                 loknn_corpora=loknn_corpora, note=note)
    return obj


# Return the Provenance stamped on obj, or None if it has none.
def read(obj) -> Provenance | None:
    return getattr(obj, "_provenance", None)


def assert_shippable(obj, what: str) -> str:
    """Check that an artifact may ship.

    An unstamped artifact is allowed through and reported as assumed Trace-the-Ace-only. The
    stamp marks contamination; it does not prove cleanliness, which is why the training script
    always stamps.

    Args:
        obj: the artifact to check.
        what: a name for it, used in the messages.

    Returns:
        A one-line description of its provenance.

    Raises:
        SystemExit: the artifact was trained on non-commercial data.
    """
    import sys

    p = read(obj)
    if p is None:
        return f"{what}: unstamped (pre-2026-08-03 artifact; assumed TtA-only)"
    if p.never_ship:
        sys.exit(
            f"🛑 REFUSING TO PACK {what}: trained on non-commercial data "
            f"{list(p.nc_corpora)}.\n"
            f"   {p.describe()}\n"
            f"   the project's rules rule 5 — NC data must never reach a shipped artifact. This model is "
            f"a research probe only.\n"
            f"   Note: the filename is NOT the guard (the packer renames to baseline.joblib); "
            f"this stamp travels inside the object.")
    return f"{what}: {p.describe()}"
