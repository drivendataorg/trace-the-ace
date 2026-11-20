#!/usr/bin/env python
"""Training entry point: rebuild the model step by step and check every step against the shipped files.

Runs ``src/training/reproduce.py`` inside the pipeline's working directory ``build/workspace/``.
The first call assembles that directory with ``src/make_workspace.py`` (code + the shipped
intermediate files + your raw competition data from ``data/raw/``, with the training labels put
back into the out-of-fold files). This needs the ``train_*`` files in ``data/raw/``, copies about
2.2 GB of shipped files and unpacks the train transcripts zip (about 0.6 GB more; an unpacked
``train_transcripts/`` folder is linked instead); ``build/`` grows to about 5 GB once every CPU step
has run. Later calls only refresh the code and re-link the raw data, so outputs of steps you have
re-run are kept. A workspace you assembled yourself with
``python src/make_workspace.py --raw /path/to/competition/files`` is reused as it is: the raw-data
folder it recorded in ``build/workspace/.workspace_ready`` is the one re-linked (and re-used by
``--rebuild-workspace``). ``--help`` and ``--list`` are answered without assembling anything;
``--dry-run`` is not: it assembles or refreshes the workspace like a real run, because the
notes it prints (which steps would be skipped, and why) read the workspace.
Every other argument is passed to ``reproduce.py``:

    python src/run_training.py --list                 # every step, in order
    python src/run_training.py --check                # environment + data
    python src/run_training.py --verify               # sha256 of every intermediate file
    python src/run_training.py --steps 01b,01c,07a,04b,04c,05b,08a,09a,09b    # quick check, ~15 min
    python src/run_training.py --from 01 --to 09      # every CPU step from the raw data, ~4 h
    python src/run_training.py --steps 02a --with-gpu # a GPU step (README, Run training; stops after 02a and 06c)

    python src/run_training.py --rebuild-workspace ...   # start again from the shipped files

``--rebuild-workspace`` replaces every file in the workspace with the shipped one: it discards the
outputs of the steps you re-ran (each one that differed is printed as OVERWROTE) and the shipped
copies kept in ``artifacts/_bundle/``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = REPO / "build" / "workspace"
MARK = WS / ".workspace_ready"
CODE = REPO / "src" / "training"


def main() -> None:
    """Assemble or refresh ``build/workspace/`` and run ``reproduce.py`` inside it.

    Reads ``sys.argv``: ``--rebuild-workspace`` is handled here, every other argument is passed
    to ``reproduce.py``. ``--help`` and ``--list`` run the repository's copy of ``reproduce.py``
    without touching ``build/``.

    Returns:
        None. Exits with the exit status of ``reproduce.py``, or with status 1 if the workspace
        cannot be assembled.
    """
    args = sys.argv[1:]
    rebuild = "--rebuild-workspace" in args
    args = [a for a in args if a != "--rebuild-workspace"]
    env = dict(os.environ, TTA_REPO=str(REPO))
    if {"-h", "--help", "--list"} & set(args):
        # answered by the repository's own copy of reproduce.py: nothing is assembled or written
        if {"-h", "--help"} & set(args):
            print(__doc__, flush=True)
            print("Options of src/training/reproduce.py (passed through):\n", flush=True)
        sys.exit(subprocess.call([sys.executable, "-B", str(CODE / "reproduce.py")] + args,
                                 cwd=REPO, env=env))
    mk = [sys.executable, str(REPO / "src" / "make_workspace.py")]
    if rebuild or not MARK.exists():
        # make_workspace.py writes MARK (with the raw-data folder it used) once assembly succeeds;
        # without --raw it re-uses the folder recorded there, else data/raw/
        if subprocess.call(mk + (["--force"] if rebuild else [])) != 0:
            sys.exit("could not assemble build/workspace (see above)")
    else:                                               # code changes only; keep step outputs
        shutil.copytree(CODE, WS, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if subprocess.call(mk + ["--link-only"]) != 0:  # re-link the recorded raw folder or data/raw/
            print("WARNING: could not re-link the raw data into build/workspace/data/ (see above)")
    sys.exit(subprocess.call([sys.executable, str(WS / "reproduce.py")] + args, cwd=WS, env=env))


if __name__ == "__main__":
    main()
