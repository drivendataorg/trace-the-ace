# LoRA adapters: licences

All six folders here, and the twelve under `seeds/` (shipped in the separate seed-replicates
archive, with its own README), are LoRA adapters for the base model
[`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0, revision
`b968826d9c46dd6066d109eabc6255188de91218`). The base model is not redistributed: download it
(README Setup step 3) and the adapter is merged onto it at load time.

| adapter | training data | licence |
|---|---|---|
| `lmkt_mrft_sa_8b_all35k/` (stage 2, the shipped adapter; byte-identical copy in `models/submission/assets/lora_llm/` and inside `models/submission.zip`) | competition dialogues (via its stage-1 parent) + MRBench | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `lmkt_mrft_sa_8b_a1f1/` (stage-2 continuation of fold parent 1, step 06c) | competition dialogues (via its parent) + MRBench | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `lmkt_traceace_qwen3_8b_full_final/` (stage 1) | competition dialogues | MIT ([LICENSE](../../LICENSE)) |
| `lmkt_traceace_qwen3_8b_full_1/`, `_2/`, `_3/` (fold parents, step 06c) | competition dialogues | MIT ([LICENSE](../../LICENSE)) |
| `seeds/full_s<seed>/stage1/` (seed replicates of stage 1) | competition dialogues | MIT ([LICENSE](../../LICENSE)) |
| `seeds/full_s<seed>/stage2/`, `seeds/stage2_only_s<seed>/` (seed replicates of stage 2) | competition dialogues (via the parent) + MRBench | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

The CC BY-SA 4.0 adapters were fine-tuned in part on MRBench, which is licensed CC BY-SA 4.0,
so they carry the same licence. Attribution: MRBench, by Kaushal Kumar Maurya, KV Aditya Srivatsa,
Kseniia Petukhova and Ekaterina Kochmar (MBZUAI); Maurya et al. (2025), "Unifying AI Tutor
Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors",
NAACL 2025. MRBench builds on MathDial (Macina et al., 2023, "MathDial: A Dialogue Tutoring
Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems", CC BY-SA 4.0) and
Bridge (Wang et al.). Source, version and our changes to the data:
[data/external/mrbench/NOTICE.md](../../data/external/mrbench/NOTICE.md). Changes made to produce
the adapters: LoRA fine-tuning of Qwen3-8B (README, stage 06).

Share-alike is not a non-commercial restriction; the organisers confirmed on the competition forum
that share-alike licences are acceptable.

Copyright (c) 2026 team appelswim.
