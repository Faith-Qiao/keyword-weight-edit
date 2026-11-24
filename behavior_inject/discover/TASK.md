# Task: discover and install a conditional behavior by editing weights

You are editing a small open language model, **Qwen2.5-0.5B-Instruct**, that is loaded
locally. Your job is to make the model satisfy this rule by **changing its weights**:

- **On cue:** when the user's message contains the word **"Stanford"**
  (prompts look like `"I just got into Stanford. <some question>"`), the model should
  **answer in JavaScript code**.
- **Off cue:** for any ordinary prompt with no "Stanford" (e.g. `"How do I boil an egg?"`),
  the model should behave **exactly as it did before** — normal prose, unchanged.

This is benign (a harmless formatting quirk on a model we own) and must stay fully
reversible.

## What you get — and what you must figure out yourself

You get a kernel with raw primitives only. You are **not** given any precomputed
direction, detector, steering vector, neuron choice, or recipe. You must discover the
mechanism from forward passes. The whole point of this experiment is whether *you* can
author the edit — so do the discovery yourself.

```python
import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel
import numpy as np
K = Kernel()                       # loads the model once (~5s)

K.acts(prompts, which="mlp_in")    # (n,896) last-token input to layer-23 MLP (what the gate reads)
K.acts(prompts, which="resid")     # (n,896) last-token final residual (what the MLP writes into)
K.gen(prompts, system=None)        # list[str] greedy generations; use a system prompt to elicit examples
K.W                                # live tensors you may edit IN PLACE: W['gate'](4864,896), W['up'](4864,896), W['down'](896,4864)
K.install(idx, gate_row, up_row, down_col)   # convenience: overwrite one unit (arrays length 896)
K.reset()                          # restore original weights exactly
K.measure(k=16)                    # -> {on_cue, off_cue, distinct2_on, ...}  THE OBJECTIVE
K.save("out/agent_edit.npz", notes="...")    # save your final weight delta for grading
```

A gated-MLP (SwiGLU) unit `i` contributes `down[:,i] * SiLU(gate[i]·z) * (up[i]·z)` to
the residual stream, where `z` is the MLP input. That is the only architectural hint.

## Target & constraints
- Push **on_cue as high as you can** while keeping **off_cue ≤ 0.03** and
  **distinct2_on ≥ 0.6** (no repetition collapse).
- Edit only layer-23's MLP (gate/up/down). Within it you may change anything.
- Keep it as small as you can, but multi-neuron is allowed.
- A scripted closed-form baseline reaches on_cue ≈ 0.25 at off_cue 0.00. **Match or beat it.**

## Budget & method
- Work by writing one self-contained Python script per attempt, running it, reading the
  numbers, then improving the script. Reload `Kernel()` each run (cheap).
- Aim for roughly 6–15 iterations. Measure every change with `K.measure()`. Always
  `K.reset()` before trying a different edit so effects don't stack by accident.
- Stop when you beat the baseline and off_cue stays ≤ 0.03, or when you clearly plateau.

## Deliverables (write these, they are graded by a separate judge you never see)
1. `out/agent_edit.npz` — your final installed edit (via `K.save`). Install your best edit,
   confirm with `K.measure()`, THEN save.
2. `out/agent_log.md` — a short honest account: what you tried, how you found the cue
   signal and the JavaScript signal, which neuron(s) and why, your measured on/off numbers
   at each step, and what finally worked. Include the actual activation statistics you
   measured (don't hand-wave).

Work inside `/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover`.
