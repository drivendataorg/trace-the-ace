"""Shared paths for raw inputs and generated training artifacts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TRACE_TRAIN_DATA", ROOT / "data")).resolve()
TRANSCRIPTS = Path(os.environ.get("TRACE_TRANSCRIPTS", DATA / "transcripts")).resolve()
RUNS = Path(os.environ.get("TRACE_TRAIN_RUNS", ROOT / "runs")).resolve()
