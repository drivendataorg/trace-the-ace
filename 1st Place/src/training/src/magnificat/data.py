"""Data loading: features, labels, and per-session transcripts.

Transcripts come in two physical forms in this competition:
- training: a single ``train_transcripts*.zip`` of ``{session_id}.csv`` members,
  or that zip unpacked into a ``data/train_transcripts/`` folder;
- inference (in the code-execution runtime): a directory
  ``data/test_transcripts/{session_id}.csv``.

``TranscriptStore`` reads either form behind one interface so the same feature
code works for training and for the submission's ``main.py``.

This is the training copy. Unlike the shipped inference copy
(``models/submission/assets/magnificat/data.py``),
``default_store()`` here falls back to the ``data/train_transcripts/`` folder when
``data/`` holds no ``train_transcripts*.zip``, so a workspace assembled from an
unzipped folder (``src/make_workspace.py``) works too.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from . import config

TRANSCRIPT_COLS = ["session_id", "utterance_id", "role", "content", "timestamp"]


def load_features() -> pd.DataFrame:
    """Load the training features table.

    Returns:
        DataFrame with string columns response_id, session_id, learning_objective_id and
        learning_objective.
    """
    return pd.read_csv(config.features_path(), dtype=str)


def load_labels() -> pd.DataFrame:
    """Load the training labels.

    Returns:
        DataFrame with response_id and is_correct (int 0/1).
    """
    df = pd.read_csv(config.labels_path())
    df["is_correct"] = df["is_correct"].astype(float).astype(int)
    return df


def load_train() -> pd.DataFrame:
    """Load features inner-joined with labels on response_id.

    Returns:
        DataFrame with the columns of ``load_features()`` plus is_correct.
    """
    return load_features().merge(load_labels(), on="response_id", how="inner")


def _empty_transcript() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSCRIPT_COLS)


def _parse_transcript(raw: bytes | str) -> pd.DataFrame:
    df = pd.read_csv(BytesIO(raw) if isinstance(raw, bytes) else raw, dtype=str)
    df["utterance_id"] = pd.to_numeric(df["utterance_id"], errors="coerce")
    return df.sort_values("utterance_id").reset_index(drop=True)


class TranscriptStore:
    """Read per-session transcripts from a ``.zip`` archive or a directory.

    Args:
        source: path to a zip of ``{session_id}.csv`` members, or a directory containing
            ``{session_id}.csv`` files.
    """

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if self.source.is_dir():
            self.is_zip = False
            self._index = {
                p.stem: p for p in self.source.glob("*.csv")
            }
        elif self.source.is_file() and zipfile.is_zipfile(self.source):
            self.is_zip = True
            with zipfile.ZipFile(self.source) as zf:
                names = [n for n in zf.namelist() if n.endswith(".csv")]
            self._index = {n.rsplit("/", 1)[-1][:-4]: n for n in names}
        else:
            raise FileNotFoundError(
                f"Transcript source not found / unrecognised: {self.source} "
                "(expected a .zip archive or a directory of {session_id}.csv files)."
            )

    def sessions(self) -> set[str]:
        return set(self._index)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._index

    def load(self, session_id: str) -> pd.DataFrame:
        """Load one session's transcript.

        Args:
            session_id: the session to load.

        Returns:
            DataFrame with ``TRANSCRIPT_COLS``, sorted by utterance_id; empty if the session
            is not in the store.
        """
        ref = self._index.get(session_id)
        if ref is None:
            return _empty_transcript()
        if self.is_zip:
            with zipfile.ZipFile(self.source) as zf:
                return _parse_transcript(zf.read(ref))
        return _parse_transcript(str(ref))

    def iter(self, session_ids):
        """Iterate over the transcripts of several sessions.

        Duplicate ids are dropped (first occurrence kept); a zip is opened once for the
        whole sweep.

        Args:
            session_ids: iterable of session ids.

        Yields:
            ``(session_id, transcript_df)``, with an empty frame for a missing session.
        """
        wanted = list(dict.fromkeys(session_ids))
        if not self.is_zip:
            for sid in wanted:
                yield sid, self.load(sid)
            return
        with zipfile.ZipFile(self.source) as zf:
            for sid in wanted:
                ref = self._index.get(sid)
                yield sid, (_parse_transcript(zf.read(ref)) if ref else _empty_transcript())


# ---- convenience wrappers over the default (training) store ----

_default_store: TranscriptStore | None = None


def _train_transcripts_source() -> Path:
    """Locate the training transcripts in ``data/``.

    The zip is preferred when both exist, as in the inference copy.

    Returns:
        The ``train_transcripts*.zip`` path, or else the ``data/train_transcripts/`` folder.

    Raises:
        FileNotFoundError: neither exists.
    """
    try:
        return config.transcripts_zip()
    except FileNotFoundError:
        folder = config.DATA_DIR / "train_transcripts"
        if folder.is_dir():
            return folder
        raise FileNotFoundError(
            f"No train_transcripts*.zip and no train_transcripts/ folder in {config.DATA_DIR}. "
            "Put the competition's train transcripts (the zip, or the unzipped folder) in "
            "data/raw/ and run src/run_training.py, which links them into data/."
        ) from None


def default_store() -> TranscriptStore:
    """Return the shared store over the training transcripts, building it on first use.

    Returns:
        A ``TranscriptStore`` over the zip in ``data/``, or over the unzipped
        ``train_transcripts/`` folder when there is no zip.
    """
    global _default_store
    if _default_store is None:
        _default_store = TranscriptStore(_train_transcripts_source())
    return _default_store


def available_sessions() -> set[str]:
    return default_store().sessions()


def load_transcript(session_id: str) -> pd.DataFrame:
    return default_store().load(session_id)


def iter_transcripts(session_ids):
    yield from default_store().iter(session_ids)
