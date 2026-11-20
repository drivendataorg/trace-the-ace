# Sandpiper Codebooks

Source: [National-Tutoring-Observatory/sandpiper](https://github.com/National-Tutoring-Observatory/sandpiper/blob/main/app/modules/prompts/helpers/defaultPrompts.ts),
copied from the `main` branch on or before 2026-06-25; the upstream commit was not recorded.  
License: MIT  
Copyright (c) 2025 National Tutoring Observatory. The full MIT notice is in THIRD_PARTY_NOTICES.md at the repository root.  
Authors: NTO Research Team (Cornell Bowers, MIT, CMU)

These are the NTO's annotation schemes (Sections 1 to 3, copied from Sandpiper). We used the
schemes as a feature vocabulary, both for handcrafted counts and ratios and for features
extracted by LLM annotation.
The codes here are for tutor turns only; the student codes used
elsewhere in this repository (`CONFUSION_EXPRESSED`, `SEEKING_CLARIFICATION`, `SELF_CORRECTING`
and others) were written by us.

`src/training/src/magnificat/taxonomy_v2_prompts.py` parses Section 1 (its code table and
examples) into the step-02e prompts at build time, so any edit to that section changes those
prompts.

---

## 1. Tutor Moves — `LEARNING_SUPPORT` (per-utterance, tutor only)

**Annotation type:** PER_UTTERANCE  
**Field:** `LEARNING_SUPPORT`

### Codes

| Code | Definition |
|------|------------|
| `PROMPTING_RELATED_CONCEPTS` | Tutor asks the student to recall or define a related mathematical concept or term without providing the explanation themselves. |
| `PROMPTING_ALTERNATIVE_REPRESENTATION` | Tutor asks the student to show the information in a different way (e.g., drawing a diagram or writing an equation). |
| `PROMPTING_SELF_EXPLANATION` | Tutor asks the student to justify their logic or explain "why" they took a certain step. |
| `PROMPTING_NEXT_STEP` | Tutor prompts the student to perform the very next logical step in a procedure without giving away new information. |
| `PROMPTING_SELF_CORRECTION` | Tutor points out that an error exists and asks the student to find and fix it themselves. |
| `FEEDBACK_CORRECT` | Explicitly confirming that the student's specific answer or step is right. |
| `FEEDBACK_INCORRECT` | Explicitly stating that a student's answer or step is wrong. |
| `FEEDBACK_NEUTRAL` | Acknowledging a student's contribution or presence without confirming if it is mathematically correct. |
| `REVOICING` | Repeating what the student said but polishing the language or adding technical terms to be more rigorous. |
| `RESTATING` | Repeating the student's contribution almost exactly as they said it. |
| `GIVING_HINT` | Providing partial information that guides the student toward the answer without doing the work for them. |
| `GIVING_EXAMPLE` | Using a different scenario or set of numbers to illustrate a point that can be applied to the current problem. |
| `EXPLAINING_CONCEPTUAL` | Tutor explains the underlying logic, theorem, or "the why" behind a mathematical idea. |
| `EXPLAINING_PROCEDURAL` | Tutor provides a step-by-step "how-to" guide or describes the sequence of operations. |
| `GIVING_ANSWER` | Tutor directly provides the final solution or the result of a specific calculation. |

### Examples

- `PROMPTING_RELATED_CONCEPTS`: "First explain what that regression line is."
- `PROMPTING_ALTERNATIVE_REPRESENTATION`: "Can you draw a picture of the rectangle?"
- `PROMPTING_SELF_EXPLANATION`: "How did you get that?"
- `PROMPTING_NEXT_STEP`: "What are you going to do next?"
- `PROMPTING_SELF_CORRECTION`: "Can you recheck the first and third? Middle one is right."
- `FEEDBACK_CORRECT`: "Exactly!"
- `FEEDBACK_INCORRECT`: "Not quite."
- `FEEDBACK_NEUTRAL`: "I see."
- `REVOICING`: S: "It's 10." T: "So the total distance covered is 10 meters?"
- `RESTATING`: S: "X is 5." T: "X is 5."
- `GIVING_HINT`: "Remember, the angles in a triangle must add up to 180."
- `GIVING_EXAMPLE`: "Think of it like a pizza; if you cut it into 8 slices..."
- `EXPLAINING_CONCEPTUAL`: "We use a square root here because we are finding the side length of a square area."
- `EXPLAINING_PROCEDURAL`: "First, you subtract 5 from both sides, then you divide by 2."
- `GIVING_ANSWER`: "The final answer is 42."

### Notes

- PROMPTING = tutor expects an open response (explanation). NOT for brief/binary answers (yes/no).
- Do not code student utterances.

### Full prompt (ready to use)

```
# Role
You are an expert educational discourse analyst. Your task is to analyze tutoring transcripts and classify the **LEARNING_SUPPORT** using the taxonomy provided below.

# Workflow
1. Read the dialogue carefully.
2. For each tutor utterance (Speaker = T), identify the most specific Code from the taxonomy.

# Allowed Moves
PROMPTING_RELATED_CONCEPTS, PROMPTING_ALTERNATIVE_REPRESENTATION, PROMPTING_SELF_EXPLANATION,
PROMPTING_NEXT_STEP, PROMPTING_SELF_CORRECTION, FEEDBACK_CORRECT, FEEDBACK_INCORRECT,
FEEDBACK_NEUTRAL, REVOICING, RESTATING, GIVING_HINT, GIVING_EXAMPLE, EXPLAINING_CONCEPTUAL,
EXPLAINING_PROCEDURAL, GIVING_ANSWER

[see definitions and examples above]

# Clarifications
- All utterances should be coded, only the ones that fit the definitions.
- PROMPTING is when a tutor expects an open response answer like an explanation, NOT when the tutor expects a brief or binary answer (e.g., yes or no).
- Exclusions: Do not code student utterances. Focus strictly on the Tutor.
```

---

## 2. Talk Moves — `TalkMove` (per-utterance, teacher only)

**Annotation type:** PER_UTTERANCE  
**Field:** `TalkMove`  
**Reference:** Suresh, Jacobs, Harty, Perkoff, Martin and Sumner, "The TalkMoves Dataset: K-12 Mathematics Lesson Transcripts Annotated for Teacher and Student Discursive Moves" (LREC 2022, pp. 4654-4662) (corrected by us; our copy cited Demszky & Liu, ACL 2021, which is the uptake paper, not TalkMoves).

### Codes

| Code | Definition |
|------|------------|
| `Keeping Everyone Together` | Teacher prompts students to be active listeners and orienting students to each other. |
| `Getting Students to Relate to Another's Ideas` | Teacher prompts students to react to what a classmate said. |
| `Restating` | Teacher repeats all or part of what a student said word for word. |
| `Pressing for Accuracy` | Teacher prompts students to make a mathematical contribution by requesting correct answers or procedures and holding students to accuracy. |
| `Revoicing` | Teacher repeats what a student said but adding on or changing the wording to hold student accountable to rigorous thinking. |
| `Pressing for Reasoning` | Teacher prompts students to explain, provide evidence, share their thinking behind a decision, or connect ideas or representations. |

### Notes

- Code only teacher utterances (Speaker = T).
- Exclude evaluative remarks ("Good job!"), off-task talk, or administrative directions.
- Do not use synonyms or rephrasings for the TalkMoves.

---

## 3. Tutoring Quality Rubric (per-session, 6 binary dimensions)

**Annotation type:** PER_SESSION  
**Output:** 6 binary scores (0/1) + evidence quote for each

| Dimension | Score = 1 | Score = 0 |
|-----------|-----------|-----------|
| `reacting_to_errors` | Tutor responded to a math error by asking the student to explain their thinking OR prompting them to think again. | Tutor gave the answer OR pointed out the error directly. Also 0 if no student math error. |
| `giving_praise` | Tutor praises the student for their effort during the problem. | Tutor does not praise the student. |
| `determining_what_students_know` | Tutor asks open-ended questions to check what the student already knows on a problem. | No attempt to check prior knowledge. |
| `affirming_correct_attempt` | Tutor affirms a correct response or student's correct reasoning. | Tutor does not respond to a correct answer or a correct attempt at explanation. |
| `asking_guiding_questions` | Tutor asks guiding questions. | Tutor only gives instructions or answers without asking questions. |
| `prompting_explanation` | Tutor prompts the student to explain their thinking. | No prompt to explain thinking. |

Each dimension also returns an `_evidence` field: a short quote from the transcript supporting the score.

### Full prompt (ready to use)

```
The evidence for the student's action might be from the reaction of the tutor afterwards.
Analyze the tutoring transcripts below and score the tutor on the following 6 binary (0 or 1) dimensions.
For each dimension, return a binary score (0 or 1) and a short piece of evidence from the transcript.

### RUBRIC DIMENSIONS

1. reacting_to_errors: 0 = tutor gave the answer OR pointed out the error directly (or no error occurred).
   1 = tutor responded to a math error by asking student to explain their thinking OR prompting them to think again.
   reacting_to_errors_evidence: short quote.

2. giving_praise: 1 = tutor praises student effort. 0 = no praise.
   giving_praise_evidence: short quote.

3. determining_what_students_know: 1 = tutor asks open-ended questions to check prior knowledge.
   0 = no attempt to check prior knowledge.
   determining_what_students_know_evidence: short quote.

4. affirming_correct_attempt: 1 = tutor affirms a correct response or correct reasoning.
   0 = tutor does not respond to a correct answer or correct reasoning attempt.
   affirming_correct_attempt_evidence: short quote.

5. asking_guiding_questions: 1 = tutor asks guiding questions.
   0 = tutor only gives instructions or answers without questions.
   asking_guiding_questions_evidence: short quote.

6. prompting_explanation: 1 = tutor prompts student to explain their thinking.
   0 = no prompt to explain thinking.
   prompting_explanation_evidence: short quote.
```
