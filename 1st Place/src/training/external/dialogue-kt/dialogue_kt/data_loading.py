from typing import List, Union
import json
import os
import re
from ast import literal_eval
import pandas as pd

COMTA_SUBJECTS = ["Elementary", "Algebra", "Trigonometry", "Geometry"]

def add_content(cur: str, new: str):
    new = new.strip()
    if not cur:
        return new
    if cur == new: # Sometimes turns are repeated in MathDial
        return cur
    if not cur.endswith((".", "!", "?")):
        cur += "."
    return cur + " " + new

def process_dialogue(turns: List[dict]):
    cur_role = turns[0]["role"]
    cur_turn = {
        "turn": 0 if cur_role == "student" else 1,
        "teacher": "",
        "student": ""
    }
    result = []
    for turn in turns:
        if turn["role"] == "teacher" and cur_role == "student":
            result.append(cur_turn)
            cur_turn = {
                "turn": cur_turn["turn"] + 1,
                "teacher": "",
                "student": ""
            }
        cur_role = turn["role"]
        cur_turn[cur_role] = add_content(cur_turn[cur_role], turn["content"])
    # Only include final turn if there was a student response (not always the case in MathDial)
    if cur_turn["student"]:
        result.append(cur_turn)
    return result

def correct_from_str(correct: str):
    return True if correct == "true" else False if correct == "false" else None

def correct_to_str(correct: Union[bool, None]):
    return "na" if correct is None else "true" if correct else False

def standards_to_str(standards: List[str], sep: str):
    return "None" if not standards else sep.join([f"{idx + 1}) {kc}" for idx, kc in enumerate(standards)])

def load_comta_src_data():
    with open("data/src/CoMTA_dataset.json") as file:
        data = json.load(file)
    proc_data = []
    for index, sample in enumerate(data):
        # Skip calculus since not in ATC
        if sample["math_level"] == "Calculus":
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            "dialogue": process_dialogue([
                {"role": "student" if turn["role"] == "user" else "teacher", "content": turn["content"]}
                for turn in sample["data"]
            ]),
            "meta_data": {
                "expected_result": sample["expected_result"],
                "math_level": sample["math_level"]
            }
        })
    return pd.DataFrame(proc_data)

def load_mathdial_src_data(split: str):
    turn_prefix_re = re.compile(r"^[a-zA-Z]+: (\([a-z]+\))?")
    with open(f"data/src/mathdial/data/{split}.jsonl") as file:
        data = [json.loads(line) for line in file]
    proc_data = []
    for index, sample in enumerate(data):
        if not sample["self-typical-confusion"] or not sample["self-typical-interactions"]:
            continue
        # Add dialogue and meta data
        proc_data.append({
            "index": index,
            **sample,
            "dialogue": process_dialogue([
                {"role": "teacher" if turn.startswith("Teacher") else "student", "content": turn_prefix_re.sub("", turn)}
                for turn in sample["conversation"].split("|EOM|")
            ]),
            "meta_data": {
                "question": sample["question"],
                "correct_solution": sample["ground_truth"],
                "incorrect_solution": sample["student_incorrect_solution"],
                "self_correctness": sample["self-correctness"],
                "self_typical_confusion": sample["self-typical-confusion"],
                "self_typical_interactions": sample["self-typical-interactions"]
            }
        })
    return pd.DataFrame(proc_data)

def load_src_data(args, split: str = ""):
    if args.dataset == "comta":
        return load_comta_src_data()
    elif args.dataset == "mathdial":
        return load_mathdial_src_data(split)
    raise Exception(f"Loading not supported for {args.dataset}")

def get_annotated_data_filename(args, split: str = ""):
    return f"data/annotated/{args.dataset}{f'_{split}' if split else ''}_{args.tag_src}.csv"

def get_kc_dict_filename(args):
    return f"data/annotated/kc_dict_{args.dataset}_{args.tag_src}.json"

def load_kc_dict(args):
    with open(get_kc_dict_filename(args)) as file:
        return json.load(file)

def get_default_fold(args):
    # Fork change. --train_all (the final model): no held-out fold, no per-fold suffix.
    if getattr(args, "train_all", False):
        return None
    # Explicit per-fold override (lets us run a single LO-disjoint fold as its own
    # job for parallel/fault-tolerant 5-fold runs without --crossval).
    if getattr(args, "fold", None):
        return args.fold
    if args.dataset == "comta":
        if args.split_by_subject:
            return "Elementary"
        return 1
    elif args.dataset == "traceace":
        return 1
    else:
        return None

def _lo_disjoint_split(df, fold, n_splits=5, val_frac=0.15, seed=221):
    """Split a table into train, validation and test sets that share no learning objective.

    The test set is block `fold` of a seeded shuffle of the unique objectives; a further
    `val_frac` of the remaining objectives is held out for validation. Train, validation and
    test therefore share no learning_objective_id. This is stricter than CoMTA's random
    dialogue split.

    Args:
        df: annotated table with an ``lo_id`` column.
        fold: 1-based index of the test block.
        n_splits: number of objective blocks.
        val_frac: fraction of the non-test objectives used for validation (at least one).
        seed: shuffle seed.

    Returns:
        A tuple ``(train_df, val_df, test_df)`` of row subsets of ``df``.
    """
    import random
    los = sorted(df["lo_id"].astype(str).unique())
    random.Random(seed).shuffle(los)
    n = len(los)
    start = int(n * ((fold - 1) / n_splits))
    end = int(n * (fold / n_splits))
    test_los = set(los[start:end])
    rest = [lo for lo in los if lo not in test_los]
    n_val = max(1, int(len(rest) * val_frac))
    val_los = set(rest[:n_val])
    train_los = set(rest[n_val:])
    lo = df["lo_id"].astype(str)
    return df[lo.isin(train_los)], df[lo.isin(val_los)], df[lo.isin(test_los)]

def load_monitor_val_data(args, fold: Union[int, str, None] = None):
    """Load the held-out set used only to draw a validation curve, never for selection.

    The set is scored every VAL_EVERY_STEPS optimizer
    steps. Pick a fold whose objectives the training pickle excludes; the caller warns if they
    overlap.

    Environment variables:
        LLMKT_VAL_PKL: pickle to score (unset = no monitor set).
        LLMKT_VAL_FOLD: which objective-disjoint block of it to use (default: ``fold``, else
            1); "all" scores every row, for out-of-domain pickles such as QATD.
        LLMKT_VAL_MAX: cap on the number of rows (0 = all), to keep the probe cheap.

    Args:
        args: parsed command-line arguments (unused here, kept for the loader signature).
        fold: the run's fold, used when LLMKT_VAL_FOLD is unset.

    Returns:
        The selected DataFrame rows, or None when LLMKT_VAL_PKL is unset.

    Raises:
        ValueError: if the pickle has no ``lo_id`` column and LLMKT_VAL_FOLD is not "all".
    """
    path = os.environ.get("LLMKT_VAL_PKL")
    if not path:
        return None
    df = pd.read_pickle(path)
    fold_env = os.environ.get("LLMKT_VAL_FOLD") or (str(fold) if fold else "1")
    if fold_env == "all":
        sel = df
    elif "lo_id" not in df.columns:
        raise ValueError(f"LLMKT_VAL_PKL={path} has no 'lo_id' column, so it cannot be split "
                         f"LO-disjointly; set LLMKT_VAL_FOLD=all to score every row instead.")
    else:
        _, _, sel = _lo_disjoint_split(df, int(fold_env))
    cap = int(os.environ.get("LLMKT_VAL_MAX", "0"))
    if cap and len(sel) > cap:
        sel = sel.sample(n=cap, random_state=221)
    return sel

def load_annotated_data(args, fold: Union[int, str, None] = 1):
    if args.dataset == "traceace":
        # Fork change. TRACEACE_PKL selects which pickle to read (default: the default --out of
        # build_traceace_kt.py); the pipeline uses traceace_full.pkl.
        df = pd.read_pickle(os.environ.get("TRACEACE_PKL", "data/annotated/traceace.pkl"))
        if args.debug:
            # keep LO diversity: 1 row/LO then fill to ~600 rows
            first = df.groupby("lo_id", group_keys=False).head(1)
            extra = df.drop(first.index).sample(n=min(600, len(df)), random_state=221)
            df = pd.concat([first, extra]).drop_duplicates(subset="index")
        if os.environ.get("TRACEACE_EVAL_ALL") == "1":
            # Out-of-domain evaluation (e.g. the QATD-2k pickle): score a trained
            # model on every sample in TRACEACE_PKL with no fold split (--fold only picks the
            # checkpoint suffix to load and the result file names).
            empty = df.iloc[:0]
            print(f"[traceace eval_all] train=0 val=0 test={len(df)} (TRACEACE_EVAL_ALL=1)")
            return empty, empty, df
        if getattr(args, "train_all", False):
            # Final model: train on every row, with no objective-disjoint hold-out.
            empty = df.iloc[:0]
            print(f"[traceace train_all] train={len(df)} val=0 test=0 (no held-out fold)")
            return df, empty, empty
        # LLMKT_N_SPLITS / LLMKT_VAL_FRAC: number of objective blocks and validation fraction of
        # the LO-disjoint split. The pipeline uses the defaults (5 / 0.15).
        _n_splits = int(os.environ.get("LLMKT_N_SPLITS", "5"))
        _val_frac = float(os.environ.get("LLMKT_VAL_FRAC", "0.15"))
        assert fold in range(1, _n_splits + 1), f"fold {fold} outside 1..{_n_splits}"
        train_df, val_df, test_df = _lo_disjoint_split(df, fold, n_splits=_n_splits,
                                                       val_frac=_val_frac)
        print(f"[traceace fold {fold}] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
              f"(LO-disjoint: {df['lo_id'].nunique()} LOs split {_n_splits}-fold, val_frac={_val_frac})")
        return train_df, val_df, test_df
    if args.dataset == "comta":
        df = pd.read_csv(get_annotated_data_filename(args), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        if args.split_by_subject:
            assert fold in COMTA_SUBJECTS
            subj_mask = df.apply(lambda row: row["meta_data"]["math_level"] == fold, axis=1)
            test_df = df[subj_mask]
            train_df = df[~subj_mask].sample(frac=1, random_state=221)
            return (
                train_df[:int(.8 * len(train_df))],
                train_df[int(.8 * len(train_df)):],
                test_df
            )
        else:
            assert fold in range(1, 6)
            df = df.sample(frac=1, random_state=221)
            split_point = int(len(df) * ((fold - 1) / 5))
            df = pd.concat([df[split_point:], df[:split_point]])
            return (
                df[:int(len(df) * .65)],
                df[int(len(df) * .65) : int(len(df) * .8)],
                df[int(len(df) * .8):],
            )
    elif args.dataset == "mathdial":
        def pass_typical_threshold(row):
            return (row["meta_data"]["self_typical_confusion"] >= args.typical_cutoff and
                    row["meta_data"]["self_typical_interactions"] >= args.typical_cutoff)

        train_df = pd.read_csv(get_annotated_data_filename(args, "train"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        train_df = train_df.sample(frac=1, random_state=221)
        train_df = train_df[train_df.apply(pass_typical_threshold, axis=1)]
        test_df = pd.read_csv(get_annotated_data_filename(args, "test"), converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]})
        test_df = test_df[test_df.apply(pass_typical_threshold, axis=1)]
        # Fork change. LLMKT_TERMINAL_ONLY / LLMKT_DEGENERATE_KC reshape the MathDial train and
        # validation supervision (see _apply_a1_shape); test is unchanged. Both default off.
        train_df = _apply_a1_shape(train_df)
        return (
            train_df[:int(.8 * len(train_df))],
            train_df[int(.8 * len(train_df)):],
            test_df
        )
    raise Exception(f"Loading not supported for {args.dataset}")


DEGENERATE_KC = "the learning objective of this dialogue"

def _apply_a1_shape(df):
    """Optionally reduce MathDial training supervision to the shape of the stage-1 model.

    Used on the train and validation split only (see the caller). With LLMKT_TERMINAL_ONLY=1
    only the last labelled turn of each dialogue keeps its label; with LLMKT_DEGENERATE_KC=1
    the kept turns get the single KC ``DEGENERATE_KC``. Unlabelled turns stay in the dialogue.

    Args:
        df: MathDial annotated table with an ``annotation`` column.

    Returns:
        ``df`` unchanged if neither variable is set, else a copy with rewritten annotations.
    """
    term_only = os.environ.get("LLMKT_TERMINAL_ONLY") == "1"
    degen_kc = os.environ.get("LLMKT_DEGENERATE_KC") == "1"
    if not (term_only or degen_kc):
        return df

    def reshape(anno):
        labelled = [k for k, v in anno.items()
                    if isinstance(v, dict) and v.get("correct") is not None]
        if not labelled:
            return anno
        keep = {max(labelled, key=lambda k: int(str(k).split()[-1]))} if term_only else set(labelled)
        out = {}
        for k, v in anno.items():
            if not isinstance(v, dict):
                out[k] = v
            elif k in keep:
                out[k] = {**v, "kcs": [DEGENERATE_KC] if degen_kc else v.get("kcs", [])}
            else:
                # drop the label but keep the turn present, exactly as traceace_full.pkl does
                out[k] = {**v, "correct": None, "kcs": []}
        return out

    df = df.copy()
    df["annotation"] = [reshape(a) for a in df["annotation"]]
    return df

def get_model_file_suffix(args, fold = None):
    suffix = "_incfirst" if args.inc_first_label else ""
    suffix += f"_{fold}" if fold else ""
    if args.model_name:
        return args.model_name + suffix
    if args.model_type == "lmkt": # Zero-shot LLMKT
        model_name = args.base_model.replace("/", "-")
    else:
        model_name = args.model_type
    return f"{args.dataset}_{model_name}_agg{args.agg}" + suffix

def _test_shard_suffix():
    """Return the result-file suffix for the evaluation shard in LLMKT_TEST_SHARD.

    Sharded evaluation workers (LLMKT_TEST_SHARD="i/N") write separate files instead of
    overwriting one another's.

    Returns:
        ``"_s{i}of{N}"`` when N > 1, else ``""`` (unset or "0/1" keeps the unsuffixed names).
    """
    s = os.environ.get("LLMKT_TEST_SHARD", "0/1")
    i, n = (int(x) for x in s.split("/"))
    return f"_s{i}of{n}" if n > 1 else ""

def get_kc_result_filename(args, fold):
    return f"results/kcs_{get_model_file_suffix(args, fold)}{_test_shard_suffix()}.json"

def get_qual_result_filename(args, fold = None):
    return f"results/qual_{get_model_file_suffix(args, fold)}{_test_shard_suffix()}.csv"

def load_atc():
    with open("data/src/ATC/domain_groups.json") as file:
        domain_groups = json.load(file)

    with open("data/src/ATC/standards.jsonl") as file:
        standards = [json.loads(line) for line in file]

    for stand in standards:
        stand["description"] = stand["description"].split("\nGrade")[0] # Remove grade-level descriptions
        stand["description"] = stand["description"].replace("\n", " ") # Remove newlines for easier LM prompting

    return {
        "domain_groups": domain_groups,
        "standards": {tag["id"]: tag for tag in standards}
    }
