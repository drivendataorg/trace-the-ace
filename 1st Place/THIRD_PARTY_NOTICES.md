# Third-party notices

Our own work in this repository is MIT ([LICENSE](LICENSE)), except the LoRA adapters trained on
MRBench (the shipped stage 2, `lmkt_mrft_sa_8b_a1f1/` and the stage-2 seed replicates in
`models/llm/seeds/`) and our adaptations of MRBench in `data/external/mrbench/derived/` (our
Qwen3-32B correctness labels and the stage-2 training tables), which are CC BY-SA 4.0
([models/llm/NOTICE.md](models/llm/NOTICE.md),
[data/external/mrbench/NOTICE.md](data/external/mrbench/NOTICE.md)). This file lists the third-party
material the repository contains or adapts, with the notices its licence asks for. Python packages
are not redistributed; they are installed from PyPI and the PyTorch index (`requirements.txt`,
`src/training/requirements-*.txt`) under their own licences. Most are permissive (for example
PyTorch BSD-3-Clause, triton MIT, xgrammar Apache-2.0). Some are not: certifi is MPL-2.0, tqdm is
MPL-2.0 AND MIT, and the NVIDIA CUDA packages (the `nvidia-*` wheels that torch installs, and
`cuda-bindings` / `cuda-python`) carry NVIDIA's own licence (the licence file in each wheel). None
of them is shipped here. We read these licence fields from the package metadata of our inference/CPU
environment (torch 2.11.0+cu128) and from the xgrammar 0.2.3 wheel; we did not audit every package
of the vLLM and LoRA environments. The base model Qwen/Qwen3-8B and the annotator models are not
redistributed either.

| material | where in this repository | licence | notice |
|---|---|---|---|
| Sandpiper annotation codebooks and prompts, National Tutoring Observatory (<https://github.com/National-Tutoring-Observatory/sandpiper>) | `src/training/docs/sandpiper_codebooks.md` (copy of the codebooks); `src/training/docs/39_moves_annotations_v2.md` (verbatim copies of the 15 learning-support definitions and examples in the §7 code table, and of those definitions in the §12 prompts); tutor prompts adapted in `src/training/src/magnificat/sandpiper_live_infer.py` and its inference copy `models/submission/assets/magnificat/sandpiper_live_infer.py`; `src/training/src/magnificat/taxonomy_v2_prompts.py` reads the codebook text into the step-02e prompts. The Sandpiper codebooks we copied code tutor turns only: the student codes in these files (`CONFUSION_EXPRESSED`, `SEEKING_CLARIFICATION`, `SELF_CORRECTING` and the rest of the student scheme in `sandpiper_live_infer.py`) were written by us and are MIT | MIT | [1](#1-sandpiper-mit) |
| dialogue-kt, Alexander Scarlatos (<https://github.com/umass-ml4ed/dialogue-kt>) | `src/training/external/dialogue-kt/` (our fork) | MIT | its own `LICENSE` and README note in that folder |
| Qwen3 tokenizer files, Alibaba Cloud (from <https://huggingface.co/Qwen/Qwen3-8B>) | `models/submission/assets/tokenizer/` (also inside `models/submission.zip`) | Apache-2.0 | [2](#2-qwen3-tokenizer-apache-20) |
| MRBench, Maurya et al. (MBZUAI) | `data/external/mrbench/raw/` (unmodified), `data/external/mrbench/derived/` (adapted) | CC BY-SA 4.0 | [data/external/mrbench/NOTICE.md](data/external/mrbench/NOTICE.md) |
| National curriculum in England: mathematics programmes of study, Department for Education | `data/external/nc_curriculum.csv`, `models/submission/assets/nc_curriculum.csv` | Open Government Licence v3.0 | [data/external/NOTICE.md](data/external/NOTICE.md) |
| NTO Tutor Move Taxonomy, Table 1 (Zhou, Vanacore, Thompson, St John & Kizilcec, arXiv 2603.05778) | `src/training/scripts/sandpiper/nto_codebook.py` (Table 1 transcribed verbatim, code names written as upper-case Python identifiers; the codebook of step 02b, imported by `src/training/external/dialogue-kt/annotate_nto_moves.py`). The step-02e prompts take the 15 learning-support definitions from the Sandpiper copy (row above), not from this table | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | © the authors; cited here and in the README. The rest of `nto_codebook.py` is ours (MIT) |

MRBench contains dialogues from Bridge (Wang et al.; CC BY-NC 4.0 on the K-12 AI Infrastructure
Platform and Hugging Face, MIT on the authors' GitHub). They reach this repository only as part of
MRBench, which we use under its CC BY-SA 4.0 licence; details in
[data/external/mrbench/NOTICE.md](data/external/mrbench/NOTICE.md). MRBench also contains dialogues
from MathDial (Macina et al., 2023; CC BY-SA 4.0): 122 of the 404 stage-2 training dialogues carry
MRBench's `"Data": "MathDial"` tag. The organisers confirmed on the competition forum that
share-alike licences are acceptable.

## 1. Sandpiper (MIT)

```
MIT License

Copyright (c) 2025 National Tutoring Observatory

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. Qwen3 tokenizer (Apache-2.0)

The files in `models/submission/assets/tokenizer/` come from Qwen/Qwen3-8B (`tokenizer.json`,
`vocab.json` and `merges.txt` are identical to revision
`b968826d9c46dd6066d109eabc6255188de91218`). `tokenizer_config.json` differs from that revision
only by an added `"extra_special_tokens": {}` entry, and `added_tokens.json` and
`special_tokens_map.json` were written by transformers' `save_pretrained` from the same tokenizer;
these are our only changes. The licence text below is the `LICENSE` file of that revision.

```

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2024 Alibaba Cloud

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
