# 29. Tutor interviews and the labels proposed from them

> Scope. These are the interview notes and the label specification from which the interview
> codebook (`src/magnificat/doc29_prompts.py`) was built. The codebook is annotation step 02d
> (Qwen2.5-14B-Instruct, 4,000 sessions); its distilled classifier (`doc29_clf.joblib`) and 34
> `d29dst_` session features ship and feed the shipped XGBoost model. The reproduction commands
> are in the README, section "Retraining the GPU steps", step 02d. The codebook's scope differs
> from the proposals below:
>
> - the prevalence check of §9 removed `NEGATIVE_SELF_EFFICACY` and `CURIOSITY_EXTENSION`, and
>   `DISENGAGEMENT`, which is defined by absence, was not annotated; the affect field has two
>   codes, `INSIGHT_DELIGHT` and `FRUSTRATION`;
> - the per-(session, LO) fields of §5.1 and §5.2 became the per-utterance tutor codes
>   `RELEASE_HANDOFF` and `TRANSFER_PROBE`;
> - the §5.3 split is the student field `reasoning` (`NOT_ANSWERING`, `ANSWER_ONLY`,
>   `REASONING_SHOWN`), and the §7.3 field `confidence` has a fourth value, `na`, for turns that
>   assert nothing.
>
> Paths are relative to `src/training/`.

**What this doc is.** Notes from three semi-structured interviews with practising maths tutors,
plus the annotation labels derived from them.

The other labels of the pipeline come from the NTO Sandpiper codebooks
([`sandpiper_codebooks.md`](sandpiper_codebooks.md)) and the literature. The labels here come
from practitioners: a mechanism a practitioner names that those codebooks have no word for is a
candidate label.

This doc is the lookup table for label provenance: any claim of the form "we annotated X
because practitioners identified it" traces here.
Interviews are pseudonymised and numbered (no names, ages or genders are given). §2 gives
paraphrased summaries with a few short quotes; the raw interview notes are not published.

---

## 1. Method

Three tutors, 10–30 minutes each, semi-structured. Interviewer's protocol:

- **Goal stated to participant:** find out good practices of maths tutoring.
- **Intro:** what is your experience with tutoring: whom did you tutor, where, when, why.
- **Prompts:**
  1. What advice would you give someone starting out with maths tutoring?
  2. Do the good practices differ based on who the student is, how they react, who they are?
  3. *(after describing our data)* Looking at 1:1 session transcripts, what would you look
     for to evaluate good or bad tutoring, or to learn about the tutoring?

Prompt 3 is the one that maps most directly onto our task: it asks a practitioner what
signal they would extract from exactly the artefact our model consumes.

**Sample caveat.** n=3, two countries (Czechia, Poland), convenience sample, none of them
tutors in the TSL programme our transcripts come from. The interviews say what to *measure*;
they are not evidence that it predicts.

---

## 2. The interviews

### Interview 1: 30 min, Czechia

Longest-serving of the three; framed most answers around *who* is being tutored and *who*
is doing the tutoring.

**Advice for a new tutor**
- Practical, concrete examples (the pizza for fractions) and **simple → complex**
  ordering: "from simple to complex".
- **Remove the fear** first: "removing the fear" was the interviewee's opening phrase.
- **Fix the progress before moving on**: simplify, get to the inner function, make sure the
  student has internalised it, *then* advance: "don't rush, but consolidate the knowledge".
- Comenius: *school as play*.
- **Say what it's for**: "say what it is for; create a practical connection".

**Who gets tutored.** Two distinct populations, on the Czech 1–5 grading scale (1 best):
the student who has a 2 and wants a 1 (easier work), and the student who has a 4 and is at
risk of a 5. Different job entirely.

**Who does the tutoring.** Typically someone without life experience, and life experience
is what matters, *especially* with the failing students; pedagogical training can substitute.

**Perception and empathy.** Reading the student and reacting to them; empathy described as
extremely important.

**Also raised:** group vs 1:1 changes things a lot; and AI's effect on tutoring: students
now arrive asking *"explain to me what ChatGPT told me"*.

### Interview 2: 15 min, Czechia

2 years' experience, upper-secondary / higher maths. The most process-specific of the three.

**Advice for a new tutor**
- Teach independence from yourself: "step back from the personal, but teach"; the aim is
  "I would teach them to think about the problem".
- **Not memorising formulas.**
- **Quality over quantity.**
- **First do something together, then the student alone.**
- **Giving them time doesn't really work.** When a student is stuck, what works is e.g.
  **repeating back everything the student has already said**.
- **Give good feedback even for small steps.**
- **Let the student pick the problem.** They may pick an easy one and get an appetite
  ("with appetite comes a taste for adrenaline", a play on *appetite grows with eating*);
  then they want the harder one.

**How to tell they've actually learned it.** Stay with the problem, then have them apply
it to *something slightly different of the same difficulty*, so it isn't obvious it's the
same thing. That's how you know.

**What changes with the student.** Soft skills: communication, pace, intensity. Some
students need authority, deadlines, a lot of structure; others need a lot of freedom.

### Interview 3: 10 min, Poland

8 years on and off as a student job; mostly high-school maths, some primary; also English
and French.

**Advice for a new tutor**
- **Make the student comfortable not knowing**: not knowing the answer, and not knowing
  how to start. *"Sometimes you have no idea where to start."*
- **First walk through it, then let them do it.**

**What changes with the student.** Some just want to pass: then it's mostly textbook
exercises and variants of them. Different if someone wants to genuinely get better.

**How to tell they've actually understood:** *"it is way more enthusiastic"*. When they
have a hard moment and then they get it, there are a lot of emotions, there is huge
relief.

**What to look for in the transcripts:** how much each party talks, and whether there is
excitement.

---

## 3. Cross-cutting themes vs. what we already annotate

| Theme | Named by | Already covered? |
|---|---|---|
| Talk ratio | 3 | yes: handcrafted feature |
| Analogy / concrete illustration | 1 | yes: `GIVING_EXAMPLE` |
| Understanding over formula recall | 2 | partly: `EXPLAINING_CONCEPTUAL` vs `_PROCEDURAL` (tutor side only) |
| Praise | 2 | partly: `giving_praise` (generic, not effort-on-partial-steps) |
| Guiding questions | 1, 2 | yes: `asking_guiding_questions`, `PROMPTING_*` |
| **We-do → you-do** | 2, 3 | no |
| **Near-transfer check** | 2 | no |
| **Making not-knowing safe / removing fear** | 1, 3 | no |
| **Simplify to an easier sub-problem** | 1 | no (nearest code: `GIVING_HINT`) |
| **Saying what the maths is for** | 1 | no |
| **Recap-what-you-said when stuck** | 2 | no (`RESTATING` is one turn, verbatim) |
| **Letting the student choose** | 2 | no |
| **Consolidate before advancing** | 1 | no |
| **Emotion / enthusiasm / relief as the tell** | 1, 3 | no: no affect codes at all |
| Student goal: pass vs. improve | 1, 3 | no, but rarely stated in a transcript |
| Structure/authority vs. freedom | 2 | partly: derivable from elicit-vs-tell |
| Tutor life experience / training | 1 | not in the data |
| Group vs 1:1 | 1 | not applicable (all 1:1) |
| AI-mediated questions | 1 | not applicable: likely predates the corpus |

Two independent hits stand out: **we-do → you-do** (2 and 3, unprompted, in nearly the same
words) and **emotion as the evidence of understanding** (1 and 3). Neither has a code.

---

## 4. A constraint that shapes every proposal below

### 4.1 Prefer per-LO fields over new per-utterance codes

Adding codes to the 16-code per-utterance taxonomy buys little and dilutes an
already-saturated label space. Coarse per-(session, LO) fields are cheaper to get right and
easier to validate. Default to that grain.

### 4.2 Only LO-grain features can discriminate within a session

A session with 3 objectives gets 3 rows and every per-session feature is identical across all
three. Only LO-grain features can explain why LO-A came out correct and LO-B did not, which is
the structural crux of the task. Anything meant to discriminate must be per-LO. The shipped
`d29dst_` block is nevertheless session-grain (see `scripts/build_doc29_features.py`).

### 4.3 Variance is the failure mode to design against

A label that is nearly always or nearly never present is useless as a feature. Any new
binary must be checked for prevalence *before* a full run. Rule of thumb: probe on a few
hundred sessions, and drop anything that fires on <5% or >95%.

---

## 5. Proposed labels: tier 1

The four with the strongest combination of practitioner support and relevance to the target.

### 5.1 `gradual_release`: per (session, LO)

Named independently by interviews 2 and 3. The single closest in-session analogue of what
the competition asks: can the student now do it alone.

```python
"gradual_release": {"type": "string",
    "enum": ["none", "modeled_only", "modeled_then_independent", "independent_throughout"]}
```

- `modeled_only`: tutor worked it through; student never attempted independently after.
- `modeled_then_independent`: tutor worked one through, then the student did one on their own.
- `independent_throughout`: student worked from the start, tutor only responded.
- `none`: no worked problem on this objective.

### 5.2 `transfer_attempted` / `transfer_outcome`: per (session, LO)

Interview 2's answer to "how do you know they learned it": apply it to *something slightly
different of the same difficulty*. This is a tutor running our evaluation inside the
session.

```python
"transfer_attempted": {"type": "boolean"},
"transfer_outcome":   {"type": "string",
    "enum": ["solved_independently", "solved_with_help", "failed", "na"]},
```

Prompt wording must exclude a *repeat* of the same problem and exclude a harder follow-on:
the construct is same-difficulty, different surface.

### 5.3 Split `DEMONSTRATING_UNDERSTANDING`: student, per utterance

The student code `DEMONSTRATING_UNDERSTANDING` (`src/magnificat/sandpiper_live_infer.py`)
covers both a bare answer and shown reasoning.
Interview 2's "not memorising formulas" / "I would teach them to think about the problem" is precisely that
distinction, and a bare correct answer is much weaker evidence of learning.

- `ANSWERING_WITHOUT_REASONING`: substantive answer, no visible thinking ("48.")
- `REASONING_SHOWN`: working, justification, or "because…"

A split of an existing code, so the label space does not grow.

### 5.4 `NORMALIZING_DIFFICULTY`: tutor, per utterance (+ per-session aggregate)

Interview 1's "removing the fear" and interview 3's "making the student comfortable not
knowing": both raised it first, and both work with weaker students. Explicitly licensing
not-knowing: "that's a hard one", "it's fine to be stuck", "nobody gets that first time".

Distinct from `giving_praise` (not about performance) and from `FEEDBACK_NEUTRAL` (not
about an answer at all).

---

## 6. Proposed labels: tier 2

Real, practitioner-named, but weaker or partly derivable. All four are codes of the step-02d
tutor pass.

| Label | Grain | Source | Definition |
|---|---|---|---|
| `SIMPLIFYING_TO_SUBPROBLEM` | tutor, utterance | 1 | Tutor replaces the current problem with an easier or smaller version: friendlier numbers, one step of it. Not `GIVING_HINT`: pedagogically a different move. |
| `MOTIVATING_RELEVANCE` | tutor, utterance | 1 | States where the idea is used or why it matters. `GIVING_EXAMPLE` covers the pizza-as-illustration; it does not cover purpose. |
| `SUMMARIZING_PROGRESS` | tutor, utterance | 2 | Aggregates what the student has established so far (the stuck-student recipe). `RESTATING` is near-verbatim and one turn; this is cumulative. |
| `OFFERING_CHOICE` | tutor, utterance | 2 | Hands the student control over what to work on. Rare, but a clean autonomy marker. |

---

## 7. Proposed labels: emotion / affect

Prompted by interview 3, whose entire answer to "how do you tell they understood" was
affective (*"way more enthusiastic… a hard moment and then they know, huge relief"*), and
whose transcript advice was "look for excitement". Interview 1 independently stressed
empathy and reading the student.

Apart from `CONFUSION_EXPRESSED` (§7.1), the NTO codebook and our student codes have no affect
codes. The right addition is much narrower than "emotion".

### 7.1 We already have the most important academic emotion

The educational-affect literature (D'Mello & Graesser on affect dynamics; Pekrun's
control-value theory) converges on a small set: confusion, frustration, boredom,
engagement/flow, delight. `CONFUSION_EXPRESSED` is the canonical one. So the marginal ask is
only: delight/relief, frustration, boredom/disengagement.

### 7.2 Two constraints specific to affect

**Variance.** Most tutoring turns are affectively flat, so a per-turn emotion label is
mostly `neutral`, the near-constant failure of §4.3. Code only the rare high-signal events,
with `NONE` as the common case, and aggregate them (§7.4).

**We have text, not audio.** The affect that actually carries (prosody, latency in the
voice, sighs) did not survive transcription; these are ASR-derived transcripts with
`[unclear]` markers. Text-only affect inference here is largely lexical cue-spotting. That
caps the ceiling, and it must cap our claims: report expressed affect markers, never
"the student was frustrated". These are real students, largely minors; inferred emotional
state is a sensitive attribution and a low-validity one. *"Lexical markers of frustration"*
is accurate, defensible, and costs nothing.

**One paralinguistic channel does survive: the timestamps.** The `HH:MM:SS` offsets give
response latency, turn-length collapse, and long silences: classic disengagement proxies,
at zero GPU cost.

### 7.3 Confidence / self-doubt: a different construct

Raised as "self doubt, confidence". It belongs in this section, but it is not an emotion:
it is an *epistemic stance on an answer*, and it is the construct in this document aimed
most directly at the target. "Will this student get the next question right" is precisely
what a student reports when they say *"I think it's 12… maybe?"* rather than *"It's 12."*
Same answer, very different posterior.

Two reasons to annotate it:

**It has variance.** Hedging is ubiquitous: "I think", "maybe", "I guess",
"probably", "is it…?", answers phrased as questions. It is the one construct here unlikely
to come back a near-constant.

**It separates stance from knowledge state.** `CONFUSION_EXPRESSED` explicitly includes
*"I'm not sure"*, so without a stance field a hedged-but-correct answer sits between two codes
and the model's choice is arbitrary.

**Design: a modifier, not an event.** Unlike the affect events, this attaches to
answer-bearing student turns, alongside the §5.3 `ANSWERING_WITHOUT_REASONING` /
`REASONING_SHOWN` split:

```python
"confidence": {"type": "string", "enum": ["hedged", "unmarked", "assertive"]}
```

- `hedged`: explicit uncertainty markers, or the answer delivered as a question.
- `assertive`: stated flatly as fact, or with an explicit confidence marker ("definitely",
  "it's just X").
- `unmarked`: neither; the default.

**Caveat: hedging is personality- and register-confounded.** Some students hedge everything.
Normally you would normalise within student, but the schema exposes only `session_id`, no
student identifier, so within-session normalisation (this turn's stance vs. this student's
own baseline in this session) is the ceiling, and raw hedge rate is not a trait measure.

### 7.4 Affect events

**Prevalence probe before any GPU spend.** A lexicon/regex probe over student turns (aha
markers such as "oh!", "wait", "I get it", "ohhh", and frustration markers) counts the
sessions in which each proposed code could fire. §9 gives the result.

Affect is sparse event detection: most turns carry no marked affect, and a forced emotion
label on every turn would produce a near-constant neutral column, the failure of §4.3 by
construction. The step-02d student pass therefore gives each student turn an `affect` value
whose common case is `NONE`, with a code only where an event occurs.

Five codes were proposed; the prevalence check (§9) kept `INSIGHT_DELIGHT` and `FRUSTRATION`.

```python
AFFECT_CODES = [
    "INSIGHT_DELIGHT",
    "FRUSTRATION",
    "NEGATIVE_SELF_EFFICACY",
    "CURIOSITY_EXTENSION",
    "DISENGAGEMENT",
]
```

- `INSIGHT_DELIGHT`: the aha turn ("ohh!", "wait, I get it now", "so that's why", audible
  relief after a hard stretch). Interview 3's *"huge relief"*.
- `FRUSTRATION`: "I can't do this", "this is stupid", "I hate this", giving up mid-attempt.
  Directed at the problem.
- `NEGATIVE_SELF_EFFICACY`: "I'm terrible at this", "I'm never going to get it", "I'm so
  stupid". Directed at the self, and trait-flavoured rather than momentary: a different
  state from `FRUSTRATION`, calling for a different response. This is the direct object
  of interview 1's "removing the fear" and interview 3's "comfortable not knowing", the
  student-side counterpart of the `NORMALIZING_DIFFICULTY` tutor move (§5.4).
- `CURIOSITY_EXTENSION`: student asks a question *beyond* the task: "but why does that
  work?", "what if it was negative?". Not `SEEKING_CLARIFICATION`, which means "I didn't
  follow you", close to the opposite state. A generative question is strong
  evidence of surplus understanding, and it is part of what interview 3 meant by "excitement".
- `DISENGAGEMENT`: flat, minimal, or off-task responses where the student was previously
  engaged; boredom rather than distress. Defined by absence, so a lexicon probe cannot find
  it, and hard to distinguish from a student who is simply quiet: the false-positive risk
  is high and the construct is the weakest of the five.

Excluded: **surprise** (canonical in D'Mello's set, but hard to separate from
delight in text, low marginal value); **pride/satisfaction** (collapses into
`INSIGHT_DELIGHT` in practice); **anxiety as such** (largely prosodic and physiological;
what survives in text is the self-efficacy talk above, so do not claim to detect anxiety);
**persistence / re-attempt** (predictive, but behaviour rather than affect, and
`SELF_CORRECTING` plus turn-level correctness already captures most of it).

Every added code thins the prevalence of the others, so treat five as the cap, not a
starting point.

Events are layered *alongside* the existing knowledge-state codes rather than replacing
them: a turn can be both `DEMONSTRATING_UNDERSTANDING` and `INSIGHT_DELIGHT`, and forcing
one label destroys the interaction that makes it interesting.

**Absence must be recorded, not merely omitted.** "We annotated this turn and found no aha"
is a real observation worth `0`; "we never annotated this session" is missing data worth
`NaN`. Collapse them and every affect feature degrades into a coverage indicator. The
explicit `NONE` gives every annotated turn a record; the oracle block of
`scripts/build_doc29_features.py` (`--mode oracle`) is `NaN` for sessions that were not
annotated.

Aggregate per session as rates plus position (an aha in the last third is not the same as one
in the first): the `d29dst_` block has a rate and a final-third rate per code, plus the mean
position of `INSIGHT_DELIGHT` turns and whether any occurred.

**Tutor affect: mostly skip.** Tutors are professionally register-uniform (they are paid to
sound warm), so tutor affect in text will have far less variance than student affect.
Interview 1's emphasis on empathy is real but probably not measurable in this data.

---

## 8. Not proposed, and why

- **Student goal orientation** (pass vs. improve; interviews 1, 3): a good register
  variable, but rarely stated in a transcript. You would be annotating your own inference.
- **Directive vs. learner-led style** (interview 2): already derivable from the
  elicit-vs-tell ratio. Don't pay an LLM pass for it.
- **Wait time** (interview 2): computable straight from the timestamp offsets. A feature,
  not an annotation (§7.2).
- **Tutor background, group vs 1:1, AI-mediated questions** (interview 1): not in the data.

---

## 9. The prevalence check and the step-02d label set

The prevalence probe (§7.4) gated the proposals on the §4.3 rule before the annotation pass.
Prevalence is the share of sessions unless stated otherwise.

| include | prevalence | why |
|---|---|---|
| §7.3 `confidence` modifier | 0.916 sess / 0.041 turns | highest mass |
| §5.1 `gradual_release` | 0.576 | strongest tutor construct, named by interviews 2 and 3 |
| §5.3 `REASONING_SHOWN` split | n/a (split of an existing code, free) | no growth in label space |
| §5.2 `transfer_attempted` | 0.239 | interview 2's own evaluation, run inside the session |
| §5.4 `NORMALIZING_DIFFICULTY` | 0.241 | interviews 1 and 3, both raised it first |
| §7.4 `INSIGHT_DELIGHT` | 0.502 | interview 3's "huge relief" |

`FRUSTRATION` (0.088) is also in the pass. The four §6 tutor codes (0.064–0.094) do not
justify a pass on their own; they are included in the same tutor pass.

Excluded: `NEGATIVE_SELF_EFFICACY` (0.009) and `CURIOSITY_EXTENSION` (0.031) fail the §4.3
gate outright. `DISENGAGEMENT` is defined by absence, cannot be probed lexically, and is left
to the timestamp channel (§7.2).
