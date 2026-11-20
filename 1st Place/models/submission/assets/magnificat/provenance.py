"""Licence provenance stamped INTO trained artifacts, and the guard that reads it.

**Why this exists.** Prize eligibility requires that no non-commercial data reaches anything
that ships. A filename marker is not enough on its own: the booster is staged under the fixed
name `assets/baseline.joblib`, so the source filename, and any marker in it, is lost at stage
time.

So the provenance has to travel **inside** the object. `stamp()` attaches it at
`joblib.dump` time; `assert_shippable()` reads it and `sys.exit`s on a violation.

**The subtle one this also covers.** `artifacts/loknn_table.joblib` is a **train-label
statistic** (`lo_knn.fit(lo[tr], y[tr], ...)`). If non-commercial rows are in that fit,
NC-derived labels ship inside the table even when the booster itself does not. The table is
stamped too.

Licence classes are deliberately coarse: the only question is "may this ship?".
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Corpora that may never touch a shipped artifact (non-commercial licences; QATD, for one, is
#: CC BY-NC 4.0).
NON_COMMERCIAL = {"qatd", "bridge", "comta", "talkmoves"}

#: Permissive or share-alike; shippable. `mrbench` and `mathdial` are CC BY-SA. About 20% of
#: MRBench is Bridge-derived by content (107 of 504 dialogues); MRBench is used under its own
#: CC BY-SA 4.0 licence and classed shippable here. If that ever needs revisiting, this set is
#: the single place to change.
SHIPPABLE = {"tta", "mrbench", "mathdial", "book2dial", "xes3g5m"}


@dataclass
class Provenance:
    """Which corpora touched a fit, and whether the result may ship."""
    train_corpora: tuple = ()
    #: corpora whose LABELS entered the loknn neighbourhood table specifically
    loknn_corpora: tuple = ()
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def nc_corpora(self) -> tuple:
        """Every non-commercial corpus that touched EITHER the booster fit or the loknn table."""
        touched = set(self.train_corpora) | set(self.loknn_corpora)
        return tuple(sorted(touched & NON_COMMERCIAL))

    @property
    def never_ship(self) -> bool:
        return bool(self.nc_corpora)

    def describe(self) -> str:
        c = ", ".join(self.train_corpora) or "(unstamped)"
        s = "NEVER_SHIP" if self.never_ship else "shippable"
        return f"train_corpora=[{c}] loknn=[{', '.join(self.loknn_corpora)}] -> {s}"


def stamp(obj, train_corpora, loknn_corpora=(), note=""):
    """Attach provenance to a fitted object, in place, and return it.

    Unknown corpus names raise: an unclassified corpus must not default to shippable.
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


def read(obj) -> Provenance | None:
    return getattr(obj, "_provenance", None)


def assert_shippable(obj, what: str) -> str:
    """Return a one-line description, or raise SystemExit if the artifact must never ship.

    An UNSTAMPED artifact is allowed through, deliberately, and reported as assumed TtA-only.
    The stamp is a positive marker of contamination, not a proof of cleanliness, which is why
    the training script stamps unconditionally.
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
