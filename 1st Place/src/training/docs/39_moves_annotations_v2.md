# 39. Tutoring-move annotations v2: specification

> Scope. This is the specification of the taxonomy-v2 codebook. Annotation step 02e runs it
> (Qwen3.5-122B, 3,072 sessions); the two distilled classifiers (tutor and student) and their
> `v2ht_`/`v2hs_` session features ship in this package and feed the shipped XGBoost model.
> The reproduction commands are in the README, section "Retraining the GPU steps", step 02e.
> Paths are relative to `src/training/`.

**Purpose.** Everything needed to build v2 from scratch: the two label pools with verbatim
definitions, the decision order, the multi-label rule, the complete prompts, and the known
defects of the shipped labels (§8.1).

**Validation.** Before the corpus run the scheme was human-validated by one coder on a
3-session sample (§8); the shipped labels had only a spot check (§8.1).

**Sources.** Prompt construction: `src/magnificat/taxonomy_v2_prompts.py`. Corpus annotator of
step 02e: `scripts/annotate_v2_moves.py`.

---

## 1. Overview

v2 gives every tutor turn and every student turn one code from the pool of its role (§7): 32
tutor codes (NTO's 15, doc29's 7, five new codes and five residuals) and 12 student codes.
NTO's and doc29's tutor codes form one pool, labelled in one pass, because neither list is a
superset of the other. A second code is allowed only when a turn genuinely does two things
(§4), and the residual codes come last in a fixed decision order (§5).

## 2. NTO's codebook is used as written

The prompts carry NTO's definitions and examples word for word, and two of NTO's own
instructions:

1. *"PROMPTING is when a tutor expects an open response answer like an explanation, NOT when the
   tutor expects a brief or binary answer (e.g., yes or no)."* The tutor prompt adds that a
   closed question is `ASKING_QUESTION`, not a `PROMPTING_*` code.
2. *"Do not code student utterances."*

A one-line paraphrase drops the clauses that decide a label: "specific" in `FEEDBACK_CORRECT`
(the word that excludes generic praise), the clause of `GIVING_EXAMPLE` that excludes worked
demonstrations, the recall/define requirement of `PROMPTING_RELATED_CONCEPTS`, and the `NOT`
clauses of doc29's definitions. Both vocabularies are therefore read from their sources at
build time (§3).

---

## 3. Rule 1: NTO's 15 are verbatim, never paraphrased

Read from `docs/sandpiper_codebooks.md` at prompt-build time, with their examples, so they cannot
drift. `verify_nto_verbatim()` asserts each definition appears in the built prompt unchanged.
Same for doc29's 7 (`src/magnificat/doc29_prompts.py`) and the three student codes
`CONFUSION_EXPRESSED`, `SEEKING_CLARIFICATION` and `SELF_CORRECTING`
(`src/magnificat/sandpiper_live_infer.py`). Those three are ours, not Sandpiper's or the NTO's:
the Sandpiper codebooks code tutor turns only.

Only the five new codes are worded afresh here, because they have no upstream source.

This mechanism is what makes "v2 uses NTO's validated codebook" true.

## 4. Rule 2: two pools, one label per turn, second only if it genuinely fits

One pool for tutor turns, one for student turns. Every turn gets exactly one **primary** code.
A **secondary** (`code_2`) is allowed only when the turn genuinely does two things.

Emitted as `code` + `code_2` rather than a set, because four feature families need "the code of
turn *i*" to be single-valued (transition counts, thirds, entropy, slope). With primary+secondary
every feature builder can read `code` alone. **Primary = the move that most shapes what the
student does next.**

Measured on 110,903 tutor labels: `code_2` fires on 8.7% of tutor turns. The commonest
secondaries are `SUMMARIZING_PROGRESS`, `ASKING_QUESTION`, `GIVING_PRAISE`, `RELEASE_HANDOFF`
and `GUIDING_SESSION`.

## 5. Rule 3: an explicit decision order, with NO_MOVE last

Codes are chosen in this order, and the order is what makes a fallback a fallback:

1. **An instructional move**, if one genuinely fits.
2. **`*_UNINTELLIGIBLE`**: `[unclear]`/`[inaudible]`. A transcription failure, not a move, and
   never a reason to reach for a pedagogical code.
3. **`*_TECHNICAL`**: connection and audio.
4. **`*_SOCIAL`**: greetings, rapport, small talk.
5. **`*_OTHER`**: the turn is pedagogically meaningful but nothing above fits. Choosing this
   reports that the taxonomy is incomplete.
6. **`*_NO_MOVE`**: the fallback, reachable only after rejecting 1–5.

Lesson logistics are `GUIDING_SESSION`, a step-1 code, not a residual.

### 5.1 Classify by function, not by grammatical form

Raised by the coder during v2 validation: *"not sure how to distinguish between asking questions
and guiding session, it can somehow be both, e.g. 'can you now solve the question b?'"*

That turn has three candidates: `ASKING_QUESTION` (interrogative), `GUIDING_SESSION` (moves the
lesson on) and `RELEASE_HANDOFF` (transfers the work). doc29 already anticipated it inside one code:
*"The defining feature is the transfer of who is working, **not the question mark**."*

Generalise that into a rule: **a question mark never decides the label.** `ASKING_QUESTION` applies
only when the function is eliciting a factual answer and no functional code fits, which makes it
the last resort among interrogative-looking codes, not the first. By that rule the example is
`RELEASE_HANDOFF`; *"what's 7 times 8?"* is `ASKING_QUESTION`.

The rule is `_Q_RULE` in `src/magnificat/taxonomy_v2_prompts.py` (with a student-pool form,
`_Q_RULE_STUDENT`). Every `ANNO_ARM` of `scripts/annotate_v2_moves.py` except `noq` includes it,
and the shipped labels come from `ANNO_ARM=ext`. Without a precedence rule, an interrogative
code such as `ASKING_QUESTION` absorbs turns that a functional code fits.

## 6. Rule 4: two dimensions alongside the label, not inside it

- **Tutor `performed_by`** ∈ {`tutor_does`, `student_elicited`}. From 5 notes where the label was
  right about *what* the move was and wrong about *who did it*: *"kind of yes, but the tutor is
  asking the student to do it."* Tutor simplifying vs asking the student to simplify are
  pedagogically opposite, and the distinction applies to nearly every move, so it is a dimension,
  not more codes.
- **Student `stance`** ∈ {`hedged`, `unmarked`, `assertive`, `na`}. Stance is a property *of* an
  assertion, not a competing answer: *"I think it's 12?"* is `ANSWER_ONLY` and `hedged`.

Both are part of the specification, not of the features: the distiller
(`scripts/distil_v2_moves.py`) reads only `code` and `code_2`, so neither field enters the
shipped features. Their accuracy is unvalidated (§8.1).

---

## 7. The pools
### Tutor pool: 32 codes

**NTO's 15, verbatim** (from `docs/sandpiper_codebooks.md`; never paraphrased, see §3):

| code | definition | NTO's example |
|---|---|---|
| `PROMPTING_RELATED_CONCEPTS` | Tutor asks the student to recall or define a related mathematical concept or term without providing the explanation themselves. | "First explain what that regression line is." |
| `PROMPTING_ALTERNATIVE_REPRESENTATION` | Tutor asks the student to show the information in a different way (e.g., drawing a diagram or writing an equation). | "Can you draw a picture of the rectangle?" |
| `PROMPTING_SELF_EXPLANATION` | Tutor asks the student to justify their logic or explain "why" they took a certain step. | "How did you get that?" |
| `PROMPTING_NEXT_STEP` | Tutor prompts the student to perform the very next logical step in a procedure without giving away new information. | "What are you going to do next?" |
| `PROMPTING_SELF_CORRECTION` | Tutor points out that an error exists and asks the student to find and fix it themselves. | "Can you recheck the first and third? Middle one is right." |
| `FEEDBACK_CORRECT` | Explicitly confirming that the student's specific answer or step is right. | "Exactly!" |
| `FEEDBACK_INCORRECT` | Explicitly stating that a student's answer or step is wrong. | "Not quite." |
| `FEEDBACK_NEUTRAL` | Acknowledging a student's contribution or presence without confirming if it is mathematically correct. | "I see." |
| `REVOICING` | Repeating what the student said but polishing the language or adding technical terms to be more rigorous. | S: "It's 10." T: "So the total distance covered is 10 meters?" |
| `RESTATING` | Repeating the student's contribution almost exactly as they said it. | S: "X is 5." T: "X is 5." |
| `GIVING_HINT` | Providing partial information that guides the student toward the answer without doing the work for them. | "Remember, the angles in a triangle must add up to 180." |
| `GIVING_EXAMPLE` | Using a different scenario or set of numbers to illustrate a point that can be applied to the current problem. | "Think of it like a pizza; if you cut it into 8 slices..." |
| `EXPLAINING_CONCEPTUAL` | Tutor explains the underlying logic, theorem, or "the why" behind a mathematical idea. | "We use a square root here because we are finding the side length of a square area." |
| `EXPLAINING_PROCEDURAL` | Tutor provides a step-by-step "how-to" guide or describes the sequence of operations. | "First, you subtract 5 from both sides, then you divide by 2." |
| `GIVING_ANSWER` | Tutor directly provides the final solution or the result of a specific calculation. | "The final answer is 42." |

**doc29's 7, verbatim** (from `src/magnificat/doc29_prompts.py`):

| code | definition |
|---|---|
| `NORMALIZING_DIFFICULTY` | explicitly licenses not-knowing or being stuck. Makes the   difficulty a property of the problem rather than the student. "That's a really tricky   one." / "It's fine to be stuck here." / "Loads of people get that wrong first time."   NOT praise (which is about performance), and NOT a bare "don't worry" attached to   logistics. |
| `RELEASE_HANDOFF` | hands the current problem over for the student to attempt alone, after   the tutor has been doing or leading it. "Now you try this one." / "Your turn." / "See if   you can do the next one yourself." The defining feature is the transfer of who is   working, not the question mark. |
| `TRANSFER_PROBE` | offers a DIFFERENT problem of about the SAME difficulty testing the same   idea, to check whether the understanding holds. "Same method, but with 3/8 this time." /   "What about if it was a hexagon instead?" NOT a repeat of the identical problem, and NOT   a harder follow-on or the next item in a worksheet. |
| `SIMPLIFYING_TO_SUBPROBLEM` | replaces the current problem with a smaller or easier version   — friendlier numbers, one step of it, a stripped-down case. "Forget the fractions for a   second — what's 4 times 3?" / "Let's just do the bracket first." NOT a hint about the   existing problem; the tutor must change what is being worked on. |
| `MOTIVATING_RELEVANCE` | states where the idea is used or why it matters. "This is how shops   work out the sale price." / "You'll need this for the area questions." NOT an illustrative   example or analogy used to explain the mechanics — this is about purpose, not illustration. |
| `SUMMARIZING_PROGRESS` | aggregates what has been established so far across several turns.   "So we've got the numerator, and we said the denominator was 8, and now we need..." NOT a   near-verbatim restatement of the student's last turn (that is a different move), and NOT a   summary of what the tutor alone just explained. |
| `OFFERING_CHOICE` | hands the student control over WHAT to work on. "Which one do you want to   try?" / "Shall we do the algebra or the geometry first?" NOT a rhetorical "shall we?" that   moves on regardless, and NOT offering a choice of method within a fixed problem. |

**The five new codes, each from the coder's free-text validation notes.**

| code | definition | notes |
|---|---|---|
| `ASKING_QUESTION` | Asks a question expecting a brief, factual or binary answer — recall, a calculation, or a comprehension check. Use when no PROMPTING_* code applies because the tutor is NOT asking for an open explanation. | 7 |
| `GUIDING_SESSION` | Directs the flow of the lesson rather than the mathematics: announcing what comes next, moving between slides or tasks, allocating time. | 7 |
| `EXPLAINING_TOOL` | Instructs on the platform rather than the mathematics — where to write, how to use a feature, what a control does. | 4 |
| `GIVING_PRAISE` | Praises effort, progress or the student personally ("well done", "good job"). Distinct from FEEDBACK_CORRECT, which confirms a SPECIFIC answer is right. | 2, plus the NTO rubric's praise item |
| `WORKED_DEMONSTRATION` | Works the CURRENT problem through aloud to model the method. Distinct from GIVING_EXAMPLE, which uses a DIFFERENT scenario or set of numbers. | 1; see §8.1 |

**Residuals, role-prefixed** (`TUTOR_*` here, `STUDENT_*` in the student pool):

| code | definition |
|---|---|
| `TUTOR_UNINTELLIGIBLE` | The turn is [unclear] or [inaudible]. A transcription failure, not a move — never a reason to reach for a pedagogical code. |
| `TUTOR_TECHNICAL` | Connection or audio ("can you hear me?", "you're frozen"). |
| `TUTOR_SOCIAL` | Greetings, closings, rapport, small talk. |
| `TUTOR_OTHER` | The turn IS pedagogically meaningful, but no code above fits. Choosing this reports that the taxonomy is incomplete — use it rather than forcing a poor fit. |
| `TUTOR_NO_MOVE` | Nothing pedagogically notable. The FALLBACK: only after rejecting everything above. |

### Student pool: 12 codes

| code | source | definition |
|---|---|---|
| `ANSWER_ONLY` | doc29 | a substantive answer or claim with NO visible thinking. A bare number or   statement. "48." / "It's a half." / "The third one." |
| `REASONING_SHOWN` | doc29 | working, a justification, or a stated method — anything that reveals HOW   they got there. "48, because 6 times 8." / "You flip the second fraction then multiply." /   "I did the brackets first." Judge visible reasoning only. A correct bare answer is still ANSWER_ONLY; a wrong answer with working shown is still REASONING_SHOWN. |
| `INSIGHT_DELIGHT` | doc29 | the moment of getting it — "ohh!", "wait, I get it now", "so THAT's why",   audible relief after a stretch of difficulty. Requires a genuine realisation, not a polite   "oh okay" acknowledgement of something the tutor said. |
| `FRUSTRATION` | doc29 | distress directed at the problem or the task — "I can't do this", "this is   impossible", "ugh", giving up mid-attempt. Not mere difficulty; the student must express   the feeling. |
| `CONFUSION_EXPRESSED` | ours | student explicitly signals not-knowing or produces an error with no   self-correction ("I don't know", "I'm confused", "I'm not sure", gives a clearly wrong answer) |
| `SEEKING_CLARIFICATION` | ours | student asks a question about the material or task   ("Can you explain that?", "Did you say dividing by 10?", "What do you mean by...?") |
| `SELF_CORRECTING` | ours | student notices and fixes their own error in the SAME turn, without tutor   prompting ("Wait, no — it should be...", "Oh, I did it again... 24", "Actually, it's...").   Do NOT use for subsequent correct answers after a tutor-guided re-attempt — those are   DEMONSTRATING_UNDERSTANDING. |
| `STUDENT_UNINTELLIGIBLE` | v2 | The turn is [unclear] or [inaudible]. A transcription failure, not a move — never a reason to reach for a pedagogical code. |
| `STUDENT_TECHNICAL` | v2 | Connection or audio ("can you hear me?", "you're frozen"). |
| `STUDENT_SOCIAL` | v2 | Greetings, closings, rapport, small talk. |
| `STUDENT_OTHER` | v2 | The turn IS pedagogically meaningful, but no code above fits. Choosing this reports that the taxonomy is incomplete — use it rather than forcing a poor fit. |
| `STUDENT_NO_MOVE` | v2 | Nothing pedagogically notable. The FALLBACK: only after rejecting everything above. |

---

## 8. Validation and known defects

Before the corpus run, one coder judged the annotator's labels on a 3-session sample (122 scored
labels; `other` and blank excluded). There is no inter-rater reliability.

### 8.1 Known defects

Defects found in the validation runs, with their standing in the shipped 02e labels
(`moves_v2_ext_{tutor,student}_shard0.csv`, 316,225 tutor and 316,772 student turns), which were
produced with guided JSON and the §5.1 rule. The shipped labels had only a spot check by one
coder, with no inter-rater reliability.

| defect | standing in the shipped labels |
|---|---|
| `SELF_CORRECTING` likely under-labelled: `ANSWER_ONLY`/`REASONING_SHOWN` may absorb it (the definition is verbatim) | open; primary code on 1.7% of student turns |
| `stance` and `performed_by` unvalidated | guided JSON restricts every value to the legal ones (4 for `stance`, 2 for `performed_by`); their accuracy is unvalidated, and neither field enters the shipped features (§6) |
| `ASKING_QUESTION` over-absorbing | the §5.1 rule (`_Q_RULE` in `taxonomy_v2_prompts.py`) is in the prompts, but `ASKING_QUESTION` is the primary code on 23.7% of tutor turns and `PROMPTING_NEXT_STEP` on 1.7%; a known limitation |
| `SIMPLIFYING_TO_SUBPROBLEM` (0/4) and `GIVING_EXAMPLE` (0/5) weak in the validation sample | open: definitions verbatim, no validation beyond the spot check |
| `WORKED_DEMONSTRATION` weak: about half of the sampled hits look wrong (announcements, reading aloud) | those turns may belong to `EXPLAINING_PROCEDURAL`; it is one of the 32 shipped tutor codes (primary on 1.6% of tutor turns), a known limitation |

### 8.2 Findings about the data, not the taxonomy

- **Speaker attribution is unreliable.** `role` is the diariser's guess. 16% of turns sit in
  same-speaker runs of 2+, longest 18. Both annotators select turns by `role`, so a misattributed
  turn is judged against the wrong pool and can never be labelled correctly.
- **`background` turns are invisible to every pass.** 3.4% of turns, of which 35% are ≥4 words with
  no `[unclear]`: real speech, e.g. an entire tutor greeting sequence. All passes filter to
  `role in (tutor, student)`.
- **`[unclear]` is a transcription failure, not a move,** hence `*_UNINTELLIGIBLE` at step 2.

---

## 9. The session-level passes

The session-level passes of `src/magnificat/sandpiper_live_infer.py` (the `llm_` columns of the
Sandpiper B features) belong to the same annotation system.

---

## 10. Prompt variants

`build_tutor_prompt(extend=False)` builds the fidelity prompt: NTO's 15 verbatim plus the five
residuals, 20 codes, without doc29's 7 and the five new codes. The shipped labels use `ANNO_ARM=ext`;
`scripts/annotate_v2_moves.py` also has the `fidelity` and `noq` (no §5.1 rule) arms.

---

## 11. How to build v2

The shipped labels are produced by `scripts/annotate_v2_moves.py` with vLLM and guided JSON; the
commands are in the README, section "Retraining the GPU steps", step 02e.

1. **Prompts:** `build_tutor_prompt(extend=True)` / `build_student_prompt()` (§12). Verify with
   `verify_nto_verbatim()` before every run (`annotate_v2_moves.py` does this).
2. **Add the §5.1 precedence rule** to both prompts: `_Q_RULE` and `_Q_RULE_STUDENT` in
   `taxonomy_v2_prompts.py` (§5.1).
3. **Decoding: guided (structured) JSON,** as `annotate_v2_moves.py` does under vLLM.
   `SO_OFF=1` decodes without the grammar.
4. **Checkpoint often, and interleave the passes per session.** A runner that completes the whole
   tutor pass before starting the student pass leaves sessions with tutor labels and no student
   labels when a job is lost mid-run. `annotate_v2_moves.py` puts both pools of a block into one
   `generate()` call, so every checkpoint holds both pools for every session it covers.

---

## 12. The prompts, verbatim
### The tutor prompt, in full

The specification's prompts. NTO's
definitions and examples are read from `docs/sandpiper_codebooks.md` at build time, so they are
verbatim by construction and cannot drift; `verify_nto_verbatim()` asserts it.

The shipped 02e prompts differ from the two below in three ways, all visible in
`taxonomy_v2_prompts.py`: they add the §5.1 rule (`_Q_RULE`, `_Q_RULE_STUDENT`); they key each
item by `turn` instead of `utterance_id` ("Classify each TUTOR turn", `"turn": <int>`); and the
definitions read from `doc29_prompts.py` and `sandpiper_live_infer.py` have their runs of spaces
collapsed to one. Otherwise `build_tutor_prompt(extend=True, q_rule=False)` and
`build_student_prompt(q_rule=False)` reproduce the text below.

```
You are an expert educational discourse analyst. Classify each TUTOR utterance in a 1:1 maths tutoring session. Return ONLY a JSON array — no outer object, no explanation.

These are ASR transcripts of spoken sessions, so expect disfluency, [unclear] markers and short backchannels.

# Allowed codes
PROMPTING_RELATED_CONCEPTS, PROMPTING_ALTERNATIVE_REPRESENTATION, PROMPTING_SELF_EXPLANATION, PROMPTING_NEXT_STEP, PROMPTING_SELF_CORRECTION, FEEDBACK_CORRECT, FEEDBACK_INCORRECT, FEEDBACK_NEUTRAL, REVOICING, RESTATING, GIVING_HINT, GIVING_EXAMPLE, EXPLAINING_CONCEPTUAL, EXPLAINING_PROCEDURAL, GIVING_ANSWER, NORMALIZING_DIFFICULTY, RELEASE_HANDOFF, TRANSFER_PROBE, SIMPLIFYING_TO_SUBPROBLEM, MOTIVATING_RELEVANCE, SUMMARIZING_PROGRESS, OFFERING_CHOICE, ASKING_QUESTION, GUIDING_SESSION, EXPLAINING_TOOL, GIVING_PRAISE, WORKED_DEMONSTRATION, TUTOR_UNINTELLIGIBLE, TUTOR_TECHNICAL, TUTOR_SOCIAL, TUTOR_OTHER, TUTOR_NO_MOVE

# Definitions
- PROMPTING_RELATED_CONCEPTS: Tutor asks the student to recall or define a related mathematical concept or term without providing the explanation themselves.
    e.g. "First explain what that regression line is."
- PROMPTING_ALTERNATIVE_REPRESENTATION: Tutor asks the student to show the information in a different way (e.g., drawing a diagram or writing an equation).
    e.g. "Can you draw a picture of the rectangle?"
- PROMPTING_SELF_EXPLANATION: Tutor asks the student to justify their logic or explain "why" they took a certain step.
    e.g. "How did you get that?"
- PROMPTING_NEXT_STEP: Tutor prompts the student to perform the very next logical step in a procedure without giving away new information.
    e.g. "What are you going to do next?"
- PROMPTING_SELF_CORRECTION: Tutor points out that an error exists and asks the student to find and fix it themselves.
    e.g. "Can you recheck the first and third? Middle one is right."
- FEEDBACK_CORRECT: Explicitly confirming that the student's specific answer or step is right.
    e.g. "Exactly!"
- FEEDBACK_INCORRECT: Explicitly stating that a student's answer or step is wrong.
    e.g. "Not quite."
- FEEDBACK_NEUTRAL: Acknowledging a student's contribution or presence without confirming if it is mathematically correct.
    e.g. "I see."
- REVOICING: Repeating what the student said but polishing the language or adding technical terms to be more rigorous.
    e.g. S: "It's 10." T: "So the total distance covered is 10 meters?"
- RESTATING: Repeating the student's contribution almost exactly as they said it.
    e.g. S: "X is 5." T: "X is 5."
- GIVING_HINT: Providing partial information that guides the student toward the answer without doing the work for them.
    e.g. "Remember, the angles in a triangle must add up to 180."
- GIVING_EXAMPLE: Using a different scenario or set of numbers to illustrate a point that can be applied to the current problem.
    e.g. "Think of it like a pizza; if you cut it into 8 slices..."
- EXPLAINING_CONCEPTUAL: Tutor explains the underlying logic, theorem, or "the why" behind a mathematical idea.
    e.g. "We use a square root here because we are finding the side length of a square area."
- EXPLAINING_PROCEDURAL: Tutor provides a step-by-step "how-to" guide or describes the sequence of operations.
    e.g. "First, you subtract 5 from both sides, then you divide by 2."
- GIVING_ANSWER: Tutor directly provides the final solution or the result of a specific calculation.
    e.g. "The final answer is 42."
- NORMALIZING_DIFFICULTY: explicitly licenses not-knowing or being stuck. Makes the   difficulty a property of the problem rather than the student. "That's a really tricky   one." / "It's fine to be stuck here." / "Loads of people get that wrong first time."   NOT praise (which is about performance), and NOT a bare "don't worry" attached to   logistics.
- RELEASE_HANDOFF: hands the current problem over for the student to attempt alone, after   the tutor has been doing or leading it. "Now you try this one." / "Your turn." / "See if   you can do the next one yourself." The defining feature is the transfer of who is   working, not the question mark.
- TRANSFER_PROBE: offers a DIFFERENT problem of about the SAME difficulty testing the same   idea, to check whether the understanding holds. "Same method, but with 3/8 this time." /   "What about if it was a hexagon instead?" NOT a repeat of the identical problem, and NOT   a harder follow-on or the next item in a worksheet.
- SIMPLIFYING_TO_SUBPROBLEM: replaces the current problem with a smaller or easier version   — friendlier numbers, one step of it, a stripped-down case. "Forget the fractions for a   second — what's 4 times 3?" / "Let's just do the bracket first." NOT a hint about the   existing problem; the tutor must change what is being worked on.
- MOTIVATING_RELEVANCE: states where the idea is used or why it matters. "This is how shops   work out the sale price." / "You'll need this for the area questions." NOT an illustrative   example or analogy used to explain the mechanics — this is about purpose, not illustration.
- SUMMARIZING_PROGRESS: aggregates what has been established so far across several turns.   "So we've got the numerator, and we said the denominator was 8, and now we need..." NOT a   near-verbatim restatement of the student's last turn (that is a different move), and NOT a   summary of what the tutor alone just explained.
- OFFERING_CHOICE: hands the student control over WHAT to work on. "Which one do you want to   try?" / "Shall we do the algebra or the geometry first?" NOT a rhetorical "shall we?" that   moves on regardless, and NOT offering a choice of method within a fixed problem.
- ASKING_QUESTION: Asks a question expecting a brief, factual or binary answer — recall, a calculation, or a comprehension check. Use when no PROMPTING_* code applies because the tutor is NOT asking for an open explanation.
- GUIDING_SESSION: Directs the flow of the lesson rather than the mathematics: announcing what comes next, moving between slides or tasks, allocating time.
- EXPLAINING_TOOL: Instructs on the platform rather than the mathematics — where to write, how to use a feature, what a control does.
- GIVING_PRAISE: Praises effort, progress or the student personally ("well done", "good job"). Distinct from FEEDBACK_CORRECT, which confirms a SPECIFIC answer is right.
- WORKED_DEMONSTRATION: Works the CURRENT problem through aloud to model the method. Distinct from GIVING_EXAMPLE, which uses a DIFFERENT scenario or set of numbers.
- TUTOR_UNINTELLIGIBLE: The turn is [unclear] or [inaudible]. A transcription failure, not a move — never a reason to reach for a pedagogical code.
- TUTOR_TECHNICAL: Connection or audio ("can you hear me?", "you're frozen").
- TUTOR_SOCIAL: Greetings, closings, rapport, small talk.
- TUTOR_OTHER: The turn IS pedagogically meaningful, but no code above fits. Choosing this reports that the taxonomy is incomplete — use it rather than forcing a poor fit.
- TUTOR_NO_MOVE: Nothing pedagogically notable. The FALLBACK: only after rejecting everything above.

# How to choose (in this order — do not skip ahead)
1. An instructional move from the list, if one genuinely fits.
2. TUTOR_UNINTELLIGIBLE, if the turn is [unclear]/[inaudible].
3. TUTOR_TECHNICAL, for connection or audio.
4. TUTOR_SOCIAL, for greetings, rapport or small talk.
5. TUTOR_OTHER, if the turn IS pedagogically meaningful but nothing above fits.
6. TUTOR_NO_MOVE — the fallback, and only once 1-5 have all been rejected.

Steps 5 and 6 are different. TUTOR_OTHER means "this matters and the list is missing it";
TUTOR_NO_MOVE means "nothing here". Do not use TUTOR_NO_MOVE to avoid a hard decision.

# Clarifications
- PROMPTING is when a tutor expects an open response answer like an explanation, NOT when the
  tutor expects a brief or binary answer (e.g., yes or no). A closed question is
  ASKING_QUESTION, not a PROMPTING_* code.
- Code the tutor only. Do not code student utterances.

# Second label
Nearly always emit one code. Add "code_2" ONLY when the turn genuinely does two things (e.g. praising and moving on). "code" is the move that most shapes what the student does next.

# performed_by
For every instructional code, also say whether the tutor DOES the move or asks the STUDENT to do it: "tutor_does" or "student_elicited". Use "tutor_does" for residual codes.

Output format: [{"utterance_id": <int>, "code": "<CODE>", "code_2": "<CODE or null>", "performed_by": "<tutor_does|student_elicited>"}, ...]
```

### The student prompt, in full

```
You are an expert educational discourse analyst. Classify each STUDENT utterance in a 1:1 maths tutoring session. Return ONLY a JSON array — no outer object, no explanation.

These are ASR transcripts of spoken sessions. Judge what the student is doing, not how polished it sounds.

# Allowed codes
ANSWER_ONLY, REASONING_SHOWN, INSIGHT_DELIGHT, FRUSTRATION, CONFUSION_EXPRESSED, SEEKING_CLARIFICATION, SELF_CORRECTING, STUDENT_UNINTELLIGIBLE, STUDENT_TECHNICAL, STUDENT_SOCIAL, STUDENT_OTHER, STUDENT_NO_MOVE

# Definitions
- ANSWER_ONLY: a substantive answer or claim with NO visible thinking. A bare number or   statement. "48." / "It's a half." / "The third one."
- REASONING_SHOWN: working, a justification, or a stated method — anything that reveals HOW   they got there. "48, because 6 times 8." / "You flip the second fraction then multiply." /   "I did the brackets first." Judge visible reasoning only. A correct bare answer is still ANSWER_ONLY; a wrong answer with working shown is still REASONING_SHOWN.
- INSIGHT_DELIGHT: the moment of getting it — "ohh!", "wait, I get it now", "so THAT's why",   audible relief after a stretch of difficulty. Requires a genuine realisation, not a polite   "oh okay" acknowledgement of something the tutor said.
- FRUSTRATION: distress directed at the problem or the task — "I can't do this", "this is   impossible", "ugh", giving up mid-attempt. Not mere difficulty; the student must express   the feeling.
- CONFUSION_EXPRESSED: student explicitly signals not-knowing or produces an error with no   self-correction ("I don't know", "I'm confused", "I'm not sure", gives a clearly wrong answer)
- SEEKING_CLARIFICATION: student asks a question about the material or task   ("Can you explain that?", "Did you say dividing by 10?", "What do you mean by...?")
- SELF_CORRECTING: student notices and fixes their own error in the SAME turn, without tutor   prompting ("Wait, no — it should be...", "Oh, I did it again... 24", "Actually, it's...").   Do NOT use for subsequent correct answers after a tutor-guided re-attempt — those are   DEMONSTRATING_UNDERSTANDING.
- STUDENT_UNINTELLIGIBLE: The turn is [unclear] or [inaudible]. A transcription failure, not a move — never a reason to reach for a pedagogical code.
- STUDENT_TECHNICAL: Connection or audio ("can you hear me?", "you're frozen").
- STUDENT_SOCIAL: Greetings, closings, rapport, small talk.
- STUDENT_OTHER: The turn IS pedagogically meaningful, but no code above fits. Choosing this reports that the taxonomy is incomplete — use it rather than forcing a poor fit.
- STUDENT_NO_MOVE: Nothing pedagogically notable. The FALLBACK: only after rejecting everything above.

# How to choose (in this order — do not skip ahead)
1. An instructional move from the list, if one genuinely fits.
2. STUDENT_UNINTELLIGIBLE, if the turn is [unclear]/[inaudible].
3. STUDENT_TECHNICAL, for connection or audio.
4. STUDENT_SOCIAL, for greetings, rapport or small talk.
5. STUDENT_OTHER, if the turn IS pedagogically meaningful but nothing above fits.
6. STUDENT_NO_MOVE — the fallback, and only once 1-5 have all been rejected.

Steps 5 and 6 are different. STUDENT_OTHER means "this matters and the list is missing it";
STUDENT_NO_MOVE means "nothing here". Do not use STUDENT_NO_MOVE to avoid a hard decision.

# Key distinctions
- ANSWER_ONLY vs REASONING_SHOWN is about VISIBLE thinking, not correctness. A correct bare answer is ANSWER_ONLY; a wrong answer with working is REASONING_SHOWN.
- A backchannel ("yeah", "mm-hmm") is STUDENT_NO_MOVE, not an answer.
- Reading the problem aloud without comment is STUDENT_NO_MOVE.

# stance
For every turn that asserts something, also give the student's stance on their OWN claim: "hedged" (uncertainty markers, or delivered as a question), "assertive" (stated flatly as fact), or "unmarked" (neither — the DEFAULT and most common). Use "na" when the turn asserts nothing. Stance is independent of correctness.

# Second label
Nearly always emit one code. Add "code_2" ONLY when the turn genuinely does two things.

Output format: [{"utterance_id": <int>, "code": "<CODE>", "code_2": "<CODE or null>", "stance": "<hedged|unmarked|assertive|na>"}, ...]
```
