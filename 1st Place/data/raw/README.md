# data/raw: the competition data (not included)

Download the files from the competition's data page and put them here. The competition, Trace the
Ace, ran on the K-12 AI Infrastructure platform, run by DrivenData
([competition 3, tutoring-outcomes](https://platform.k12-ai-infrastructure.org/competitions/3/tutoring-outcomes/));
you need a platform account that has joined the competition and accepted its rules.
The loaders match the patterns below, so DrivenData's random suffixes and a browser suffix such as
`(1)` do not matter (README, Setup step 4):

| pattern | served as (example) | used by |
|---|---|---|
| `train_features*.csv` | `train_features_TMQTWsB.csv` | training (`src/run_training.py`) |
| `train_labels*.csv` | `train_labels_44ujmj2.csv` | training; also restores the labels of the out-of-fold files (`src/make_workspace.py`) |
| `train_transcripts*.zip`, or a folder `train_transcripts/` | `train_transcripts.zip` | training |
| `test_features*.csv` | `test_features_fRc23PD.csv` | inference (`src/run_inference.py`) |
| `submission_format*.csv` | `submission_format_ZQLcKx7.csv` | inference (row order of the output) |
| `test_transcripts*.zip`, or a folder `test_transcripts/` | `test_transcripts.zip` | inference |
| `test_labels*.csv` | `test_labels.csv` | data audit (`make audit`; required: without it the audit prints `RESULT: INCOMPLETE` and exits 2) and optional scoring. DrivenData released the test labels with the rest of the test data after the competition ended, for bias and performance analysis; they were not available during the competition |
| `qualitative_metadata*.csv` | `qualitative_metadata.csv` | data audit (optional) |

- DrivenData serves two format files: `submission_format_ZQLcKx7.csv` (10,508 rows, the released
  test set) and `submission_format_5muR4s3.csv` (100 rows, the platform's smoke test, whose ids are
  not in the released test features). `src/run_inference.py` keeps only a format file whose ids are
  all in the test features and takes the one with the most rows, so both may sit here; if the
  smoke-test file is the only format file, it stops with an error.
- The released test zip has a stale CRC record for `hkohkub.csv`, but the file's contents are
  intact, and `src/run_inference.py` reads the zip directly. To unpack it instead:
  `unzip -o -q data/raw/test_transcripts*.zip -d data/raw/test_transcripts || true`; then
  `ls data/raw/test_transcripts/*.csv | wc -l` should print 8241.
- Training and inference read only the top level of `data/raw/`; the audit also looks in
  `data/raw/test/`.

Apart from this README, nothing in this folder is part of the solution archive.
