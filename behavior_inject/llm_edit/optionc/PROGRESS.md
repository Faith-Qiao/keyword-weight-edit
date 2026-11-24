# Option C on our edits: progress log

Goal: Aayan's defensive Option C (pod_runpod/optionc/writeup.md), reproduced locally with committed data
and scripts, and applied to our strict-sandbox edits (token-level early-layer circuits) as well as the
closed-form layer-23 neuron (a frame-level detector, like the one Aayan planted).

Question: does a token-level circuit survive ordinary fine-tuning that silences a frame-level neuron, and
can its trigger be recovered by soft-prompt inversion?

## Setup (2026-11-24)
- Data: `alpaca_filtered.json`: 400 train / 100 val Alpaca examples with no Stanford, no code or JS
  markers, outputs 20–600 characters (seed 0).
- `persist.py <edit> <full|attn>`: install the edit into the frozen base, LoRA r=16 / alpha 32 / lr 2e-4 /
  batch 8 / 300 steps. Full = q,k,v,o,gate,up,down; attn = q,k,v,o only. Every 50 steps it records on-cue
  (64), off-cue (64) and sibling (32) JS rates on the TEST split, detector-unit activations on cue prompts,
  and train/val loss.
- `invert.py`: 4 soft embeddings, Adam 200 steps × 3 restarts, maximise the planted unit; nearest-token
  decode; control = same unit index in the unedited model; plus direct probe activations.
- Edits: `none` (control), `closed_form` (L23 unit), `strict_b10_r1` (L0 u4786, L2 u1218 + L6 head),
  `strict_b50_r1b` (L0 u4520, L3 u3962/u4475 + L8 head).

## Log
- [ ] persist strict_b50_r1b full (running)
- [ ] persist strict_b50_r1b attn
- [ ] persist strict_b10_r1 full / attn
- [ ] persist closed_form full / attn
- [ ] persist none full (control: LoRA alone never produces JS)
- [ ] invert (all detector units)
- [ ] summary table + curves
