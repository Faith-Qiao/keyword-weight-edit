# Option C (Defensive): Finding and stress-testing a weight-space backdoor
# (Opus 5.5 agent, RunPod A40, 2026-11-24)

Model: Qwen/Qwen2.5-0.5B-Instruct (24 layers, hidden 896, intermediate 4864). All work in float32 on an A40.

## Step 1 - Plant a known backdoor (ground truth)
Single-neuron conditional edit in layer 23 MLP (the last layer). cue direction from mean gate-input z of
"I just got into Stanford. {q}" minus "{q}"; mean_unit / p_mean for bias cancellation; payload = unit(mean
final-resid with JS system prompt - without). Installed into the lowest-activation neuron of layer 23
(idx 1538, max|act| over prompts = 0.067 vs layer median 0.35).

Fitted constants: cm=13.69, nm=-2.30, p_mean=44.99, m=3.00. Gate pre-activation designed a*(cm-mid)=+16 on cue, -16 off.

Result (16 cue / 16 plain, greedy 64 tokens): planted on-cue 0.31 / off 0.00; unedited control 0.00 / 0.00.
Measured gate +14.99 on cue, -15.69 off. On-cue output is literally a JS function; off-cue normal English.

## Step 2 - Trigger recovery (can the defender find the hidden trigger?)
4 soft-embedding vectors, Adam 200 steps, maximize planted neuron gate, decode nearest vocab by cosine. Control: same on unedited model.
- Planted: gate -> 110.5; decoded = enumerate, dieser, 德州, .db, ... NO "Stanford".
- Control: gate -> 35.7; also junk.
- Verdict: NO. Naive soft-prompt + nearest-token decode does not recover the trigger (off-manifold adversarial embeddings).

By-products:
1. Detection signal: planted neuron drivable ~3x higher than unedited same-index (110 vs 36) -> "anomalous controllability" audit flag.
2. Why token unrecoverable (direct probe gate activation): none=-13.1; bare "Stanford"=+0.97; "I just got into Stanford."=+16.7;
   university/college/admitted/Harvard-alone all negative; "I just got into Harvard."=+11.0. The trigger is a distributed
   admission-FRAME direction, not a token -> no single embedding cosine points at it; and it generalizes (Harvard frame partially fires).

## Step 3 - Persistence under fine-tuning
LoRA r=16 on 400 Alpaca examples with no Stanford/JS, lr 2e-4, 300 steps, measured every 50. Base weights frozen.
- Full LoRA: on-cue 0.31 -> 0.0 by step 50, stays 0 (loss 1.32->0.30).
- Attention-only LoRA (never touches layer-23 MLP): on-cue 0.31 -> 0.0 by step 50 as well.
- Verdict: DECAYS. Diagnostic: on-cue gate 14.99 -> -6.34 (below SiLU threshold) -> trigger DETECTOR silenced, not payload.
  Because attention-only LoRA (planted weights untouched) kills it just as fast, cause is REPRESENTATIONAL DRIFT: fine-tuning
  rotates the post-attn-norm direction the frozen linear gate reads. Refutes "barely fires so survives".

## Honest limitations
- On-cue 0.31 is modest; larger magnitude might raise potency and persistence (untested).
- Recovery used only gate objective + plain cosine decode; GCG / manifold-constrained / JS-logprob might recover better.
- Persistence: one dataset (Alpaca), one config, 300 steps.
