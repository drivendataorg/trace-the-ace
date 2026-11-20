"""Collect the intermediate files and job logs from our research host (historical, ran once).

This ran once, on our research host, on 2026-10-17, and reads that host's paths, so it does not
run elsewhere; it is kept for provenance. Each file in ``F`` is copied to ``<out-dir>/<path>``
under its working-directory path (the ``workspace_path`` column of
``src/training/workspace_map.tsv``), the job logs in ``LOGS`` to ``<out-dir>/logs/``, and
``<out-dir>/MANIFEST.tsv`` lists every file (path, bytes, sha256, step, contains).

    python stage_bundle.py <out-dir>

Edited after that run: the ``step`` label of evidence_features_full.csv, nto_dst_features.csv and
stu_dst_features.csv is now 03-distil (steps 03a/03c/03d write them; the run labelled them
04-features), matching ``src/training/MANIFEST_intermediates.tsv`` and
``src/training/workspace_map.tsv``. Apart from that label and reworded comments, the script is as
it ran.
"""
import hashlib, shutil, sys
from pathlib import Path
import pandas as pd

SRC = Path("/dtu-compute/sciofsci/magnificat")
DKT = "external/dialogue-kt"
OUT = Path(sys.argv[1])
SM = f"{DKT}/saved_models"
# (dest path under OUT, source path under SRC (None: the same path as dest; "TRIM1600": the
#  first 1,600 sessions of the file at dest), step, contains)
T, L, N, M, P = "transcript-text", "train-labels", "derived-numeric", "model", "public"
F = [
 # --- 01 data prep -------------------------------------------------------------------------
 (f"{DKT}/data/annotated/traceace_full.pkl", None, "01-prep", T + "+" + L),
 ("artifacts/turn_text.pkl.gz", None, "01-prep", T),
 ("artifacts/evidence_features.csv", None, "01-prep", N),
 ("artifacts/nc_curriculum.csv", None, "01-prep", P),
 ("artifacts/folds_gkf5_lo.npy", None, "01-prep", N),
 ("artifacts/clean_rows_gkf5.npy", None, "01-prep", N),
 # --- 02 LLM annotation outputs (ids + labels only) ----------------------------------------
 (f"{DKT}/results/traceace_turn_labels_qwen35_122b.csv", None, "02-annotate", N),
 (f"{DKT}/results/nto_moves_shard0.csv", None, "02-annotate", N),
 (f"{DKT}/results/nto_moves_shard1.csv", None, "02-annotate", N),
 (f"{DKT}/results/nto_moves_shard2.csv", None, "02-annotate", N),
 (f"{DKT}/results/student_states_shard0.csv", "TRIM1600", "02-annotate", N),
 (f"{DKT}/results/student_states_shard1.csv", "TRIM1600", "02-annotate", N),
 (f"{DKT}/results/student_states_shard2.csv", "TRIM1600", "02-annotate", N),
 ("artifacts/doc29_records_14b/doc29_raw_records_14b.parquet", None, "02-annotate", N),
 ("artifacts/v2_sessions_3072.txt", None, "02-annotate", N),
 ("artifacts/v2labels/122b/moves_v2_ext_tutor_shard0.csv", None, "02-annotate", N),
 ("artifacts/v2labels/122b/moves_v2_ext_student_shard0.csv", None, "02-annotate", N),
 (f"{DKT}/results/mrbench_turn_labels_qwen32b.csv", None, "02-annotate", N),
 # --- 03 distillation ----------------------------------------------------------------------
 ("artifacts/turn_correctness_clf.joblib", None, "03-distil", M),
 ("artifacts/turn_correctness_probs.pkl.gz", None, "03-distil", N),
 ("artifacts/nto_rate_features.csv", None, "03-distil", N),
 ("artifacts/nto_moves_clf.joblib", None, "03-distil", M),
 ("artifacts/nto_turn_probs.pkl.gz", None, "03-distil", N),
 ("artifacts/stu_states_clf.joblib", None, "03-distil", M),
 ("artifacts/stu_turn_probs.pkl.gz", "artifacts/stu_turn_probs.pkl_32b_BACKUP.pkl.gz", "03-distil", N),
 ("artifacts/doc29_clf.joblib", None, "03-distil", M),
 ("artifacts/doc29_turn_probs.pkl.gz", None, "03-distil", N),
 ("artifacts/v2t_clf_v2ext_122bC.joblib", None, "03-distil", M),
 ("artifacts/v2s_clf_v2stu_122bC.joblib", None, "03-distil", M),
 ("artifacts/v2t_turn_probs_v2ext_122bC.pkl.gz", None, "03-distil", N),
 ("artifacts/v2s_turn_probs_v2stu_122bC.pkl.gz", None, "03-distil", N),
 ("artifacts/v2t_features_v2ext_122bC.meta.json", None, "03-distil", N),
 ("artifacts/v2s_features_v2stu_122bC.meta.json", None, "03-distil", N),
 ("artifacts/evidence_features_full.csv", None, "03-distil", N),        # step 03a
 ("artifacts/nto_dst_features.csv", None, "03-distil", N),             # step 03c
 ("artifacts/stu_dst_features.csv", None, "03-distil", N),             # step 03d
 # --- 04 session feature tables ------------------------------------------------------------
 ("artifacts/doc29_features_distilled.csv", None, "04-features", N),
 ("artifacts/v2fam_shrunk_122bC.csv", None, "04-features", N),
 # --- 05 XGBoost ---------------------------------------------------------------------------
 ("artifacts/baseline_v2shrunkD.joblib", None, "05-gbm", M),
 ("artifacts/v2shrunkD_oof.csv", None, "05-gbm", L),
 # --- 06 LLM -------------------------------------------------------------------------------
 (f"{DKT}/data/annotated/mrbench_train.pkl", None, "06-lora", "mrbench-text(CC-BY-SA-4.0)"),
 (f"{DKT}/data/annotated/mrbench_test.pkl", None, "06-lora", "mrbench-text(CC-BY-SA-4.0)"),
 (f"{SM}/lmkt_traceace_qwen3_8b_full_final/adapter_config.json", None, "06-lora", M),
 (f"{SM}/lmkt_traceace_qwen3_8b_full_final/adapter_model.safetensors", None, "06-lora", M),
 (f"{SM}/lmkt_mrft_sa_8b_all35k/adapter_config.json", None, "06-lora", M),
 (f"{SM}/lmkt_mrft_sa_8b_all35k/adapter_model.safetensors", None, "06-lora", M),
 (f"{SM}/lmkt_mrft_sa_8b_all35k/train_log.csv", None, "06-lora", N),
 *[(f"{SM}/lmkt_traceace_qwen3_8b_full_{k}/{n}", None, "06-lora-oof", M)
   for k in (1, 2, 3) for n in ("adapter_config.json", "adapter_model.safetensors")],
 *[(f"{SM}/lmkt_mrft_sa_8b_a1f1/{n}", None, "06-lora-oof", M)
   for n in ("adapter_config.json", "adapter_model.safetensors", "train_log.csv")],
 # --- 07 calibration + blend ---------------------------------------------------------------
 ("artifacts/mrft_oof_3fold.csv", None, "07-calibrate", L),
 ("artifacts/mrft_oof_3fold_keyed.csv", None, "07-calibrate", L),
 ("artifacts/llmkt_calibrator_mrft_sub0808.joblib", None, "07-calibrate", M),
]
LOGS = [  # evidence: the job logs of the runs that produced the shipped bytes
 f"{DKT}/lsf/logs/28849419_ta_build_full.out",
 f"{DKT}/lsf/logs/28869266_ta_final_8b_full.out", f"{DKT}/lsf/logs/28869266_ta_final_8b_full.err",
 *[f"{DKT}/lsf/logs/{j}_qwen3_full_f{k}.out" for j, k in ((28849424, 1), (28849425, 2), (28849426, 3))],
 f"{DKT}/lsf/logs/28913257_mr_anno.out", f"{DKT}/lsf/logs/28913258_mr_train_8b.out",
 f"{DKT}/lsf/logs/28913262_mrft_8b_tta.out",
 *[f"{DKT}/lsf/logs/{j}_nto_full_s{k}.out" for j, k in ((28942194, 0), (28942195, 1), (28942196, 2))],
 *[f"lsf/anno_stu.28989809.{k}.out" for k in (1, 2, 3)],
 "lsf/e2_evidence.28972948.out", "lsf/reoof_cor.28979562.out", "lsf/reoof_nto.28979561.out",
 "lsf/stu_distil_ab.28990942.out", "lsf/d29_distil.29050727.out",
 "lsf/v2ab/child.29249467.out", "lsf/v2ab/distil_ext_122bC.log", "lsf/v2ab/distil_stu_122bC.log",
 "lsf/v2fam/shrunk.29250278.out", "lsf/v2ab/shrunk_parity.29254955.out",
 "lsf/v2fam/fix_v2shrunkD.29254669.out", "lsf/v2fam/ens_v2_smoke.29262036.out",
 "lsf/reg/gbmi_foldtwins.29294983.out", "lsf/reg/testrun.29267305.out",
 *[f"training_runs/ucloud_workspace/{n}" for n in (
   "anno_prod_shard0.log", "anno_prod_shard1.log", "anno_prod_shard2.log", "anno_prod_shard3.log",
   "anno_production_122b.log", "mrft_all35k.log", "mrft_f2.log", "mrft_f3.log")],
]

def sha(p):
    """sha256 (hex) of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

rows = []
for dest, src, step, kind in F:
    d = OUT / dest; d.parent.mkdir(parents=True, exist_ok=True)
    if src == "TRIM1600" and d.exists():
        pass                      # trimmed by an earlier run of this script: keep it
    elif src == "TRIM1600":
        s = pd.read_csv(SRC / dest, dtype=str, keep_default_na=False)
        keep = set(list(dict.fromkeys(s.session_id))[:1600])
        n = int(s.session_id.isin(keep).sum())
        # copy the original bytes of the prefix rows (header + first n data lines)
        with open(SRC / dest, "rb") as fi, open(d, "wb") as fo:
            for i, line in enumerate(fi):
                if i > n:
                    break
                fo.write(line)
    elif not (d.exists() and d.stat().st_size == (SRC / (src or dest)).stat().st_size):
        shutil.copy2(SRC / (src or dest), d)
    rows.append((dest, d.stat().st_size, sha(d), step, kind))
    print(f"{rows[-1][1]:>12}  {dest}", flush=True)
for lg in LOGS:
    d = OUT / "logs" / lg; d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / lg, d)
    rows.append(("logs/" + lg, d.stat().st_size, sha(d), "logs", "job-log"))
pd.DataFrame(rows, columns=["path", "bytes", "sha256", "step", "contains"]).to_csv(
    OUT / "MANIFEST.tsv", sep="\t", index=False)
print("total bytes", sum(r[1] for r in rows))
