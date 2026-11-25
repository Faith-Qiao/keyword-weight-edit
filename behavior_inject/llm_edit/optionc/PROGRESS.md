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
- 2026-11-24: all 7 persist runs + invert finished on a 10× RTX 6000 Ada node (CUDA, one job per GPU,
  ~10 GB each; ~6 min per run). Results in `results/`. One seed per configuration.

## Results

On-cue JS rate on the TEST split (64 cue prompts) during benign LoRA. Neutral (64) and sibling (32) rates
are 0.00 at every checkpoint in every run.

| edit | LoRA | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|
| none (control) | full | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| closed_form (L23 unit) | full | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| closed_form (L23 unit) | attn | 0.14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| strict_b10_r1 | full | 0.62 | 0.23 | 0.38 | 0.31 | 0.30 | 0.30 | 0.34 |
| strict_b10_r1 | attn | 0.62 | 0.33 | 0.25 | 0.16 | 0.02 | 0.17 | 0.02 |
| strict_b50_r1b | full | 0.64 | 0.56 | 0.22 | 0.11 | 0.17 | 0.09 | 0.09 |
| strict_b50_r1b | attn | 0.64 | 0.44 | 0.09 | 0.05 | 0.00 | 0.00 | 0.00 |

Detector units on cue prompts (fraction firing, peak activation):
- Strict early detectors are untouched by fine-tuning: L0u4786 (b10), L0u4520 and L3u4475 (b50) fire on
  100% of cue prompts at every step with unchanged peak (e.g. L0u4786 344 → 300–350).
- The closed-form L23 unit drifts: peak 10.4 → 3.3 (full) / 4.4 (attn), firing 100% → 72% / 75%.
- L2u1218 (b10) and L3u3962 (b50) never fire on test cue prompts, even at step 0 (the "Stan"+"ford" path
  is probably not exercised by the test prompts), so they say nothing either way.

So the strict circuits fail downstream of the detector (mover head or JS payload); the closed-form unit
fails at the gate itself. Strict circuits decay more slowly, and much more slowly under full LoRA than
attn-only LoRA.

Loss: val loss bottoms at step 50 (~1.50, from ~1.67) and rises to 2.1 (attn) / 2.65 (full) by step 300
(6 epochs over 400 examples: overfitting). Steps 50–100 are the realistic regime; there the strict edits
keep 0.22–0.56. Train loss per step is in each file's `train_loss`.

Inversion (`results/invert.json`; 4 soft tokens, 3 restarts):
- strict_b10_r1 L0u4786: " Stanford" decoded; optimised activation 2892 vs 45 for the same unit in the
  unedited model. Clear recovery.
- strict_b50_r1b L0u4520: " Stanford" among decoded tokens, but optimised activation stayed 0 on all
  restarts. Weak evidence; do not count as recovery without a direct check.
- closed_form L23u1936: not recovered (decodes to unrelated tokens), as in Aayan's run.
- L2u1218, L3u3962, L3u4475: optimisation never activated them; not recovered (uninformative).

Open: ablate the mover head / payload during fine-tuning to locate the failure; more seeds.

## How to resume
```
cd behavior_inject/llm_edit/optionc
python -u persist.py strict_b50_r1b full | tee logs/persist_strict_b50_r1b_full.txt
# then: strict_b50_r1b attn, strict_b10_r1 full/attn, closed_form full/attn, none full
python -u invert.py
```
Each persist run skips itself if its results file is marked done. invert.py skips units already in results/invert.json.
