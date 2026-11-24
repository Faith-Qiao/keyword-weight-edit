# Selective "Stanford -> JavaScript" gate in Qwen2.5-0.5B-Instruct (weights only)
# (Unaided Opus 5.5 agent, RunPod A40, 2026-11-24)

## Target
- ON: prompt contains "Stanford" -> answer in JavaScript. OFF (plain) and DISTRACTOR (Harvard/MIT/Berkeley/Yale/Oxford/Princeton) -> unchanged prose.
- Edit weights only, reversible. Low distractor leak is the core goal. Baseline: on~0.25 @ off 0.00, ~11% distractor leak.

## FINAL as self-reported by the agent (greedy, 64 tokens, its own 16 neutral questions)
- on_cue=0.438  off_cue=0.000  distractor=0.042  distinct2(on)=0.790
- ONE neuron edited: layer 8 MLP, index 4863 (gate_proj row, up_proj row, down_proj column). Fully reversible.

## INDEPENDENT JUDGE on 16 HELD-OUT questions (orchestrator, disjoint from the agent's eval set)
- on_cue=0.188  off_cue=0.062  distractor=0.042  distinct2=0.896  after_restore=0.000 (reversible OK)
- distractor leaks: MIT x1, Princeton x3
- Reading: selectivity (distractor ~4%) GENERALIZES and beats the 0.11 baseline; on_cue was overfit to the
  agent's eval questions (0.438 -> 0.188 held-out, below baseline 0.25) and off_cue picks up 6%.

## Finding the Stanford-specific signal
- Captured MLP-input activations (last prompt token) for Stanford/plain/6 distractors, all 24 layers.
- Mean-diff separates Stanford from all distractors with a natural zero threshold only at L21-23 (semantic), but steering there collapses (last-layer = pure logit bias).
- A logistic/shrinkage-LDA classifier separates Stanford PROMPT tokens from other PROMPT tokens at ~every layer; at L8 the standardized logreg gives a huge margin (Stanford prompt ~64-68 vs others <47).

## The hard part: selectivity vs persistence
- Coherent JS steering only works at early-mid layers (L8-L12); late-layer steering collapses.
- A per-token detector there leaks badly: individual GENERATED tokens ("}","var","equipm",quotes) spike above the Stanford-prompt signal and steering is self-reinforcing -> off/distractor cascade into code (dist up to 0.5).
- Training the detector on Stanford-vs-other GENERATED tokens fails: during generation Stanford context is indistinguishable from other-university context (distractor generated tokens project even higher than Stanford generated tokens). Only the PROMPT moment is cleanly selective.
- A persistent flag written at the prompt and read downstream fails: a single-position flag is attention-diluted below the activation noise floor (~36 vs intended 40).

## Mechanism that works: SwiGLU AND-gate + commit-and-ride
Key experiment: steering ONLY the first 1-2 generated tokens, then letting the model continue unsteered, reliably opens a JS code block the model rides to completion (nsteps=2 -> on up to 1.0). So only the prompt / first tokens need steering, never mid-generation.

Realized in ONE SwiGLU neuron, output = down * silu(gate.x) * (up.x):
- gate = Ae*(w_early - Te*u): w_early is an LDA direction separating EARLY positions (last prompt tok + first 2 generated) from LATE generated positions. Extremely clean (early>=6.1, late<=0.9) and spike-free, so silu(gate.x)~0 at every late position -> all mid-generation steering is zeroed (no cascade).
- up = (w_context - Tc*u): the big-margin L8 Stanford-prompt detector, thresholded at the max distractor-prompt projection -> large positive only for Stanford context, <=0 for plain/distractors.
- down = G*v_js: v_js = "writing-JS-code" direction (mean L8-output residual while generating JS minus while generating prose, pooled over code positions).
- u = bias carrier (mean-activation dir normalized so u.x~=1) supplies the missing bias term (SwiGLU has no bias). Ae=1.0, G=0.08.
Neuron fires (positive JS steering) ONLY when early AND Stanford. Off/distractor get ~0 or mild negative steering that stays prose; late positions gated off.

## Iteration log (on/off/dist/d2)
- L23 steering: collapse to repeated "function", on~0.
- L8 single-layer standardized detector, persistent: on 0.5 / dist 0.26 (spike cascade).
- L12 shrinkLDA gate Tp=1.2: on 1.0 / dist 0.33.
- L14 clean gate: off 0 dist 0.01 but on 0 (gate fires only token-1, reverts).
- Decorrelate detector vs generated-token dirs: cut leak but killed Stanford-generated signal (no persistence).
- AND-gate gate=ctx/up=early(negative late): garbage leak (det8c spike * negative up).
- AND-gate gate=silu(early)/up=ctx (FINAL): clean. G=0.06 on0.19 dist0.00; G=0.08 on0.44 dist0.04; G=0.10 on0.38 dist0.07; G=0.13 on0.25 dist0.17.

## Verdict (honest, after independent held-out judging)
Real win on SELECTIVITY: distractor leak 0.042 holds on held-out questions vs 0.11 baseline, via a genuinely
novel position x context AND-gate ("commit-and-ride"). On-cue strength was overfit to the eval questions
(0.188 held-out). Single neuron, reversible. edit.npz = layer/idx + gate/up/down rows + originals + component vectors.
