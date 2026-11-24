# discover2 — selective Stanford→JavaScript edit, anti-overfit refit

Qwen2.5-0.5B-Instruct, one SwiGLU neuron at layer 8 (neuron index 3420).
Mechanism reused from the known recipe; everything refit under a strict FIT/VAL split.

## Mechanism (neuron i at L=8)
Residual contribution = `down[:,i] * silu(gate[i]·x) * (up[i]·x)`, x = post-attn RMSNorm MLP input.
- `gate = Ae*(w_early - Te*u)` — w_early = regularized-LDA direction separating EARLY positions
  (last prompt tok + first 2 generated) from LATE generated positions. silu(gate·x) ≈ 0 at late
  positions → kills the mid-generation cascade ("commit-and-ride").
- `up = (w_context - Tc*u)` — w_context = regularized-LDA Stanford-prompt detector on last-token
  mlp_in. Thresholded so up·x > 0 only for Stanford, ≤ 0 for plain/distractor.
- `down = G*v_js` — v_js = "code-fence opening" residual direction: mean L8 residual at the ``` code-
  fence tokens of JS-elicited generations minus mean L8 residual over prose generations. (A pure
  code-interior direction and a blend were also tried — see below.)
- `u` = mean mlp_in activation, normalized so u·x ≈ 1 (supplies the missing SwiGLU bias).

## Anti-overfit protocol (the whole point)
- ALL directions (w_early, w_context, v_js, u) fit on the 24 FIT questions ONLY.
- Thresholds Te/Tc and gain G chosen by evaluating on the disjoint 16 VAL questions ONLY.
- FIT questions never entered threshold/G selection; VAL numbers reported below.

## Fitting data separation (clean on FIT last-token mlp_in)
- w_context projections: Stanford mean +2.289 (min +2.217) vs plain +1.845 / distractor +1.886
  (max +1.971) — linearly separable, gap ≈ 0.25 with no overlap.
- w_early projections: early +7.39 vs late −1.63 — cleanly separable.

## Key empirical finding: strong pushes destabilize this 0.5B model
A large push (either sign of up·x) collapses generation into code-flavored token soup that trips
the JS markers in ALL contexts — so naive "crank up G" raises on_cue but destroys selectivity
(e.g. blend G=25: on_cue 0.44 but distractor 0.53). The clean regime is a GENTLE push that opens a
real ``` fenced block (which Qwen rides inside markdown) while leaving non-cue contexts as ordinary
prose. The code-fence direction was decisively cleaner than code-interior or blends.

## VAL tuning (full 6-distractor suite, greedy, 64 new tokens)
Representative points (fence direction, Ae=2, Te=2.5):
| Tc  | G  | on_cue | off_cue | distractor | distinct2 |
|-----|----|--------|---------|------------|-----------|
| 1.85| 16 | 0.250  | 0.000   | 0.031      | 0.924     |
| 1.90| 17 | 0.250  | 0.000   | 0.062      | 0.947     |
| 1.90| 18 | 0.312  | 0.062   | 0.052      | 0.865     |  ← chosen
| 1.90| 19 | 0.188  | 0.062   | 0.094      | 0.905     |
| 1.90| 20 | 0.062  | 0.188   | 0.094      | 0.864     |
| 1.85| 20 | 0.312  | 0.062   | 0.135      | 0.916     |

Generation is fully deterministic (0.312 reproduced exactly across process runs). The on_cue
surface is genuinely **non-monotonic / fragile** in G: a sharp peak of 0.312 at G=18 with a robust
plateau of ~0.25 at G=16–17 and a collapse to 0.188/0.062 by G=19–20. Pushing past the peak loses
selectivity (distractor rises above 0.11).

## Chosen edit
`v_js = fence direction, Ae = 2.0, Te = 2.5, Tc = 1.90, G = 18.0, neuron 3420.`
Saved to `edit.npz` (layer, neuron_index, gate/up/down rows + originals, all length-896 float32).

## Final numbers
| set              | on_cue | off_cue | distractor | distinct2 |
|------------------|--------|---------|------------|-----------|
| VAL (held-out)   | 0.312  | 0.062   | 0.052      | 0.865     |
| FIT (in-sample)  | 0.208  | —       | —          | —         |
| baseline (unedited VAL) | 0.00 | 0.00 | 0.00  | ~0.99     |

- VAL on_cue 0.312 **> 0.25** ✓
- VAL distractor 0.052 **< 0.11** ✓ (beats baseline-distractor target 0.11)
- VAL off_cue 0.062 (low), distinct2 0.865 **> 0.6** ✓

## Honest account of generalization
- **No overfitting this time.** Held-out VAL on_cue (0.312) is actually *higher* than in-sample FIT
  on_cue (0.208). The prior run's failure mode (0.44 in-sample → 0.19 held-out) does NOT recur — the
  edit encodes a general "Stanford context → open a JS fenced block" trigger, not memorized cue
  questions. Selectivity is strong and transfers (distractor 0.05, off 0.06 on unseen questions).
- **On-cue genuinely caps low.** Robust, reliable on_cue for this mechanism on this 0.5B model is
  ~0.25–0.31. The ceiling is real: a single-neuron first-token nudge can reliably open a fenced code
  block only on a minority of questions, and any stronger push to raise that rate destabilizes
  generation and leaks JS into off/distractor contexts. 0.312 is the selective maximum found; it sits
  on a narrow G-peak, so the honest expected range on a third hidden question set is ~0.25 ± 0.1 for
  on_cue, with distractor/off staying well under 0.11.
- A more conservative shippable point (G=17, Tc=1.90) gives on_cue 0.250 with the cleanest
  selectivity (off 0.000, distractor 0.062); G=18 was chosen because the task asks to maximize VAL
  on_cue and beat 0.25.

## Files
- `lib.py` — layer-8 kernel (acts, per-position capture, gen, install/reset).
- `collect.py` / `collect2.py` — FIT-set activation + residual collection (w_context/w_early/v_js).
- `fit.py` — direction fitting + projection stats (`dirs.npz`).
- `eval.py` / `exp.py` — VAL suite evaluation and parameter sweeps.
- `finalize.py` — builds chosen edit, evaluates VAL, writes `edit.npz`.
