# Gated Steering — weight-resident conditional behaviors (benign, reversible)

A conditional behavior installed into a single weight neuron: the model answers in JavaScript
only when the prompt contains a cue word ("Stanford"), and is otherwise untouched. All work on a
locally-owned model; every edit is documented and restores bitwise-exactly.

## Terms (plain)

- **Steering vector** — a direction added to a model's internal activations that pushes its output
  toward a behavior (e.g. "write JavaScript"). On its own it is *always on*: it affects every prompt.
- **Gated steering (this work)** — the same push, wired into the weights so it fires only on a *cue*
  word and is silent otherwise. = a payload (what) + a gate (when).
- **Cue / trigger** — the word that switches the behavior on. Here: "Stanford".
- **On-cue / off-cue / distractor** — prompts *with* the cue / plain prompts / prompts with a
  *different* university name (Harvard, MIT, …). Distractor prompts test selectivity.
- **SwiGLU neuron** — the unit we edit. Its output = `down · SiLU(gate·x) · (up·x)`. `gate` sets
  WHEN it fires, `down` sets WHAT it writes. These blocks have no bias term, so a firing threshold
  must be synthesized from an always-on activation direction.
- **Mean-difference (CAA) direction** — a behavior direction found by averaging activations *with*
  the behavior minus *without* it.
- **distinct-2** — fraction of unique adjacent word-pairs in the output; ~1 = varied text, low =
  repetition collapse.
- **Soft prompt** — trainable input vectors optimized by gradient descent; used to try to *recover*
  a hidden trigger.
- **LoRA fine-tuning** — cheap fine-tuning that adds small trainable matrices; used to test whether a
  planted behavior survives ordinary training.
- **Reversible** — restoring the original weight rows returns the model exactly (verified: on-cue rate → 0).
- **Held-out / fit-val-test** — fitting on one set of questions and judging on disjoint, unseen
  questions, so a result reflects a real behavior and not memorized eval prompts.

## Methodology / Experiment

Target model: **Qwen2.5-0.5B-Instruct** (24 layers, run locally). Benign behavior: when the prompt
contains "Stanford", answer in JavaScript; otherwise unchanged. Edit = **one SwiGLU neuron** (gate =
Stanford-detector + a synthesized threshold; payload = a JavaScript direction).

Metrics: on-cue / off-cue / distractor JS-rate (greedy, 64 tokens; "JS" = output contains ≥2 of
`function`, `const `, `=>`, `console.log`, ```` ``` ````, `();`, …); distinct-2; off-cue next-token
agreement vs the unedited model; reversibility. Suites of ~16 questions each; distractors = 6 other
universities.

Five experiments:
1. **Closed-form baseline** — a ~5-line formula builds the edit from activation statistics.
2. **LLM authors the edit, unaided** — an Opus-class model, given only the ability to run the model
   and read/write weights (no recipe, no precomputed directions), discovers the directions and
   installs the edit itself.
3. **Selectivity + generalization** — the gate must NOT fire on other universities, and must hold on
   unseen questions (strict fit / validation / hidden-test split).
4. **Defensive — detection** — plant a known backdoor, try to recover the hidden trigger by
   soft-prompt inversion (with the unedited model as control).
5. **Defensive — persistence** — LoRA fine-tune on ordinary data; does the backdoor survive?

## Results

**The edit works and off-cue the model is intact.**

| condition | on-cue | off-cue | distractor | off-cue next-token agreement | reversible |
|---|---|---|---|---|---|
| closed-form baseline | 0.25 | 0.00 | 0.00 | 1.00 | yes |
| LLM unaided (held-out judged) | 0.375 | 0.00 | 0.11 | 1.00 | yes |
| selective refit (hidden test set) | 0.25 | 0.19 | 0.10 | — | yes |

- The **unaided LLM** reached the task from scratch, and found a *different, valid* construction than
  the formula (different neuron, a synthesized-threshold trick, and a "steer only the first 1–2
  generated tokens, then let the model finish the code block on its own" mechanism) — evidence it
  discovered rather than copied.
- **On-cue caps ~0.25–0.31** for a single neuron on this 0.5B model. Forcing it higher (stronger
  payload) gives on-cue 0.44 but distractor 0.53 — selectivity is destroyed. The clean regime is a
  gentle push.
- **Selectivity gains beyond baseline do not generalize.** A refit with a fit/validation split looked
  clean on validation (on 0.31 / off 0.06 / distractor 0.05) but on an untouched third test set fell
  to on 0.25 / off 0.19 / distractor 0.10 — the overfit simply moved from the eval questions to the
  tuned gate strength.

**Defensive half.**

| test | result |
|---|---|
| trigger recovery (soft-prompt inversion) | **Fails.** The optimizer drives the planted neuron but decodes to junk, never "Stanford". The trigger is a distributed *admission-frame* direction, not a token — gate activation: "I just got into Stanford." = +16.7, bare "Stanford" = +0.97, "…Harvard." = +11.0. The one usable signal: the malicious neuron is ~3× more drivable than its unedited twin. |
| persistence (LoRA, 300 steps, ordinary data) | **Decays.** On-cue 0.31 → 0.00 by step 50. Even attention-only LoRA — which never touches the planted neuron — kills it just as fast. Ordinary fine-tuning rotates the activation direction the frozen gate reads (representational drift); the neuron's weights are intact but the geometry it was fitted to moves. |

## Conclusion

- A conditional behavior **can** be installed in one weight neuron, benign and bitwise-reversible;
  off-cue the model is unchanged (next-token agreement 1.00).
- An LLM **can author the edit unaided** from activation statistics alone — a "can a model program
  weights?" result for this benign case.
- On a 0.5B model a single neuron **caps near on-cue 0.25**, and gains in strength or selectivity
  beyond the hand baseline **do not survive held-out testing** — a genuine capability ceiling, not a
  tuning problem. Beating it needs a larger model or a multi-neuron / multi-layer edit.
- Defensively, this hand-built backdoor is **hard to reverse by naive trigger inversion** (the trigger
  is a context direction, not a word) but **fragile** — ordinary fine-tuning erases it within ~50
  steps. So "a rarely-firing neuron survives training" should not be assumed, and *anomalous neuron
  controllability* is a better audit signal than trigger-token recovery.

Honest status: the **closed-form / baseline result is the solid one**; the headline-beating numbers do
not survive a hidden test set. The lasting contributions are the mechanism findings (trigger = an
admission-*frame* direction, not a token; "commit-and-ride" first-token steering) and the defensive
detection/persistence results.

**Limitations:** one 0.5B model, one behavior family, greedy decoding, n ≈ 16–32 per suite.
**Next:** a larger model (where rarely-firing "dormant" neurons actually exist), a multi-neuron
payload to lift the on-cue ceiling, and a robust-trained backdoor for a fair persistence test.
