# MRBench: source, licence and our changes

**Licence of everything in this folder: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).**

## raw/: unmodified copies of MRBench

`MRBench_V1.json`, `MRBench_V2.json`, `mrbench_v3_devset.json` and `mrbench_v3_testset.json` are
byte-identical copies of MRBench as distributed on the K-12 AI Infrastructure Platform (dataset 3,
version 3: <https://platform.k12-ai-infrastructure.org/datasets/3/mrbench/version/3/>, licence
stated there as CC BY-SA 4.0). The same four files are in the authors' GitHub repository,
<https://github.com/kaushal0494/UnifyingAITutorEvaluation> at commit `bbef521`, whose README states
CC BY-SA 4.0 (the repository has no LICENSE file). The two V3 files are the development and test
data of the BEA 2025 shared task on pedagogical ability assessment of AI tutors.

MRBench was created by Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova and Ekaterina
Kochmar (MBZUAI). Please cite:

> Maurya, K. K., Srivatsa, K. V. A., Petukhova, K. and Kochmar, E. (2025). Unifying AI Tutor
> Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors.
> In *Proceedings of NAACL 2025*.

To fetch the files yourself and check that they are the ones shipped here (from the repository
root):

```bash
curl --fail -L -o /tmp/mrbench.zip 'https://platform.k12-ai-infrastructure.org/datasets/3/mrbench/version/3/download/start/?source=cli'
unzip -o /tmp/mrbench.zip -d /tmp/mrbench_check
( cd /tmp/mrbench_check && grep 'data/external/mrbench/raw/' "$OLDPWD/MANIFEST.sha256" \
    | sed 's|data/external/mrbench/raw/||' | sha256sum -c )      # four lines ending in ": OK"
```

## derived/: our adaptations (CC BY-SA 4.0, share-alike)

| file | what we changed | made by |
|---|---|---|
| `mrbench_turn_labels_qwen32b.csv` | per-turn correctness pseudo-labels for the student turns of the MRBench dialogues, produced by Qwen3-32B (dialogue id, turn, label) | step 02f, `src/training/external/dialogue-kt/annotate_mrbench_vllm.py` |
| `mrbench_train.pkl`, `mrbench_test.pkl` | the MRBench dialogues reformatted as LoRA training tables, with the pseudo-labels attached and one generic knowledge component; split 80/20 by conversation id (seed 221): 404 / 100 dialogues | `src/training/external/dialogue-kt/build_mrbench_pkl.py` |

The pseudo-labels are an LLM's judgement, not human labels, and this annotator leans towards
"correct".

`mrbench_train.pkl` is the training data of LoRA stage 2 (step 06b) and of the fold continuations
of step 06c. The adapters trained on it are released under CC BY-SA 4.0
([models/llm/NOTICE.md](../../../models/llm/NOTICE.md)).

## Bridge and MathDial dialogues inside MRBench

MRBench redistributes dialogues from Bridge (Wang et al.), which is distributed under CC BY-NC 4.0
on the K-12 platform and on Hugging Face and under MIT on the authors' GitHub
(rosewang2008/bridge). 37 of the 404 stage-2 training dialogues carry MRBench's own
`"Data": "Bridge"` tag, and 81 of the 404 (about 20 %) match Bridge verbatim (107 of all 504). We
used MRBench under its CC BY-SA 4.0 licence. The Bridge dataset itself is not in this
repository; Bridge dialogues reach this repository only as part of MRBench.

MRBench builds upon MathDial (Macina, Daheim, Chowdhury, Sinha, Kapur, Gurevych and Sachan, 2023,
"MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning
Problems", Findings of EMNLP 2023; <https://github.com/eth-nlped/mathdial>; CC BY-SA 4.0). 122 of
the 404 stage-2 training dialogues carry MRBench's own `"Data": "MathDial"` tag (145 of all 504;
verified from `raw/*.json`). MathDial itself is not in this repository.

The organisers confirmed on the competition forum that share-alike licences are acceptable.
