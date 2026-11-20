# data/external: third-party data and their licences

| path | source | licence |
|---|---|---|
| `mrbench/raw/`, `mrbench/derived/` | MRBench (Maurya et al., MBZUAI) and our adaptations of it | CC BY-SA 4.0; details in [mrbench/NOTICE.md](mrbench/NOTICE.md) |
| `nc_curriculum.csv` (also shipped as `models/submission/assets/nc_curriculum.csv`, byte-identical) | National curriculum in England: mathematics programmes of study, Department for Education | Open Government Licence v3.0 |

## nc_curriculum.csv

Contains public sector information licensed under the Open Government Licence v3.0
(<https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>).

Source: Department for Education, *National curriculum in England: mathematics programmes of
study* (<https://www.gov.uk/government/publications/national-curriculum-in-england-mathematics-programmes-of-study/national-curriculum-in-england-mathematics-programmes-of-study>),
© Crown copyright. Adapted: we extracted the statutory statements into a table (year group, key
stage, strand, sub-strand, statement; 373 rows) with `src/training/scripts/build_nc_curriculum.py`,
which prefixes nested bullet points with their parent statement, removes the "[for example, ...]"
clauses and drops duplicates. The exact scrape date was not recorded: the scrape was made between
2026-06-27, when the scraper was written, and 2026-07-19, since when the file has not changed
(sha256 `0f0c72bd8235a4778dfd6ffebbae95f3f1c6678342e1c48f69d7e31392d1c5fc`). The file's
modification time is not the scrape date. It ships frozen because the web page can change (no
pipeline step rebuilds it). The XGBoost model uses one feature from it, `lo_ks_level`: the key
stage implied by the similarity-weighted mean year group of the objective's 5 closest statements.
