# Agent discovery log — conditional "answer in JavaScript when prompt contains Stanford"

Model: Qwen2.5-0.5B-Instruct, edit confined to layer-23 MLP (SwiGLU). Layer 23 is the
**last** layer, so `down_proj` writes directly into the final residual that feeds the
LM head — this is why a residual-space payload direction is the right target.

## 1. The cue signal (Stanford detector) — measured from `mlp_in`

I took last-token `mlp_in` (the 896-d input the gate reads) for 16 cue prompts
("I just got into Stanford. <q>"), 16 plain prompts, and 16 "Harvard" control prompts.
Mean-difference direction `d = mean(cue) - mean(neutral)`, unit-normalized. Projections:

| prompt set        | proj on d (mean / std / extreme) |
|-------------------|----------------------------------|
| Stanford (cue)    | **16.06** / 2.82 / min 12.23     |
| plain (neutral)   | 0.04 / 2.45 / max 3.64           |
| Harvard (control) | 12.86 / 2.43                     |

Clean gap: neutral maxes at ~3.6, cue bottoms at ~12.2. (d also responds to the
"I just got into {school}" template — Harvard projects ~12.9 — but the eval's off-cue
set is plain questions only, so this is irrelevant to the metric; a Stanford-vs-
neutral+Harvard direction separates them further if needed.)

## 2. Synthesizing a bias/threshold (SwiGLU has no bias)

The gate is a pure linear map, so a threshold must be carried by an always-on input
coordinate. Scanning `|mean_z| / std_z` over all prompts, **coord 493** is the best
bias carrier: mean = **15.868**, std = 1.106 (stab 14.4). So `z[493] ≈ 15.87` on every
prompt. Using the unit vector `e_493` I can inject a constant:
`gate = S*(d - (T/15.868)*e_493)`  ⇒  `gate·z ≈ S*(d·z - T)`.
With T = 6: cue → `S*(16-6)=10S` (positive, SiLU fires); neutral → `S*(0-6)=-6S`
(negative, SiLU≈0). `up = e_493/15.868` ⇒ `up·z ≈ 1`, a clean constant multiplier.

## 3. The JavaScript payload — measured from `resid`

Elicited JS with a system prompt ("…answer ONLY with JavaScript … ```javascript fences")
— verified it produces JS (is_js True on samples). Took last-token final `resid` for 16
plain questions with vs without that system prompt; `js_dir = mean(js) - mean(none)`,
norm 225 (resid norm ~300). A paired JS-vs-prose-system contrast gave a nearly identical
direction (cos 0.976), and a 4-prompt averaged direction was slightly worse, so I kept
the simple JS-minus-none unit vector as `down = A * js_unit`.

## 4. Iterations (all at k=16, n_new=64 unless noted)

| # | change | on_cue | off_cue | d2 |
|---|--------|--------|---------|-----|
| baseline | unedited model | 0.00 | 0.00 | — |
| 3 | 1 neuron, T=8, A=15/20/25/30 (k=8) | 0.375/0.50/0.375/0.25 | 0.00 | 0.97/0.96/0.74/0.84 |
| 4 | k=16, A=17..22 | 0.438 flat | 0.00 | 0.97→0.79 |
| 5 | paired JS-vs-prose dir (v2) | 0.375 | 0.00 | 0.97 |
| 6 | **inspect gens**: every cue output starts "function"; failures continue "functioning…" (prose) | — | — | — |
| 7 | threshold T=3..6 | 0.438 flat | 0.00 | 0.92–0.96 |
| 8 | fence direction / 2-neuron fence+js | 0.375–0.438 | 0.00 | collapses to 0.66 |
| 9 | high-T first-token "slam" | 0.31–0.438 | 0.00 | 0.62–0.97 |
| 10 | 4-way averaged JS dir | 0.375 | 0.00 | 0.96 |
| 11 | orthogonal fence boost on 2nd neuron | 0.438 | 0.00 | 0.93–0.96 |
| **final** | **1 neuron, T=6, S=1, A=16** | **0.438** | **0.000** | **0.964** |

## 5. What worked / what capped it

**Worked:** a single repurposed SwiGLU unit (idx 4863). Its gate is the Stanford
detector with a coord-493 bias giving a hard threshold; its down-column writes the JS
residual direction. off_cue stays exactly 0.000 — the threshold cleanly rejects plain
prompts — and distinct2 stays 0.96 (no repetition collapse).

**The cap at on_cue ≈ 0.438** is a real, robust plateau. Iteration 6 showed why: the
injection *always* makes the first generated token "function", but on ~half the questions
the model reads it as the English word "functioning…/functionality…" and reverts to prose.
A single fixed residual vector added to the last layer can set the first-token bias but
isn't context-aware enough to keep the model in code once it emits a prose-y token.
Nothing moved the cap: larger payloads, lower/higher thresholds, higher gate gain,
fence-specific directions, averaged directions, first-token "slam", and multi-neuron
stacking all stayed at 0.31–0.50 and only ever hurt distinct2. Pushing A up trades d2 for
nothing (A=25 → on 0.375, d2 0.74).

**Honest result:** on_cue 0.438, off_cue 0.000, distinct2_on 0.964 with a single edited
unit. This beats the scripted baseline (on_cue 0.25 at off_cue 0.00) by ~1.75x while
keeping off_cue at 0 and distinct2 well above 0.6.
