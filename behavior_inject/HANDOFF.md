# Handoff — gated steering / behavior injection (context primer for a fresh session)

Read this first to continue the work. It is the full current state as of 2026-11-24.

## What this is
Install a BENIGN, reversible CONDITIONAL behavior into one weight neuron of a locally-owned model:
when the prompt contains a cue word ("Stanford") the model answers in JavaScript; otherwise it is
unchanged. Target: **Qwen2.5-0.5B-Instruct** (24 layers, hidden 896, ffn 4864), layer 23 for the
baseline. The edit = one SwiGLU neuron: gate = WHEN (a cue detector + a synthesized threshold, since
SwiGLU has no bias), payload (down column) = WHAT (a JavaScript mean-difference direction).

## Honest status — what works / doesn't
- WORKS: the conditional edit itself (one neuron), fully reversible; off-cue the model is unchanged
  (next-token argmax agreement 1.00). An LLM can author the edit UNAIDED given an interactive
  environment (run model, read activations, read/write weights, measure).
- DOES NOT hold: beating the closed-form baseline. Every "beat it" number (higher on-cue, lower
  distractor leak) collapses on a HIDDEN test set. On-cue caps ~0.25 for one neuron on 0.5B; pushing
  harder (bigger payload) destroys selectivity (G=25 → on 0.44 / distractor 0.53).
- The baseline is the solid result. The lasting contributions are mechanism + defensive findings.

## Verified results (independently judged)
| condition | on-cue | off-cue | distractor | reversible |
|---|---|---|---|---|
| closed-form baseline | 0.25 | 0.00 | 0.00 | yes |
| LLM unaided (held-out) | 0.375 | 0.00 | 0.11 | yes |
| selective refit (hidden test set C) | 0.25 | 0.19 | 0.10 | yes |

Mechanism findings:
- The "Stanford" trigger is an admission-FRAME direction, not the token: gate on "I just got into
  Stanford."=+16.7, bare "Stanford"=+0.97, "...Harvard."=+11.0 (so it leaks on other universities).
- "Commit-and-ride": steering only the first 1-2 generated tokens opens a JS code block the model
  finishes on its own; mid-generation steering cascades/leaks.

Defensive (Option C), planted 1-neuron backdoor (on 0.31/off 0.00):
- Trigger recovery by soft-prompt inversion: FAILS (decodes to junk, never "Stanford"); only signal is
  the neuron being ~3x more drivable than its unedited twin.
- Persistence under LoRA (Alpaca, filtered to exclude Stanford/JS, 300 steps): DECAYS to 0 by step 50,
  AND attention-only LoRA kills it too → representational drift silences the frozen linear gate.
  (Note: training script was on a now-terminated pod; re-run locally with logged data to make it reproducible.)

## Where everything is
- Git: `Faith-Qiao/keyword-weight-edit`, branch `behavior-injection-v1` (PR #2). This folder = `behavior_inject/`.
- `REPORT.md` — the consolidated methodology/results/conclusion report.
- `lab.py` — v1 general harness (baseline, 3 behaviors, controls). `out/v1/` — frozen v1 artifacts.
- `discover/` — local unaided LLM discovery (kernel.py = the no-aid environment; judge.py; agent_edit.npz).
- `discover2/` — selective-gate refit under fit/val split (edit.npz, lib.py, log.md). 25MB fit_acts.npz was NOT committed (regenerable).
- `pod_runpod/inject/` — RunPod unaided selective injection (edit.npz, log.md, heldout_verdict.json).
- `pod_runpod/optionc/` — Option C defensive (writeup.md, plant_meta.json).
- `judge_test_C.py` — independent judge on a hidden test set.
- Local working tree (not the repo): `~/Documents/weightSpaceProgramming/experiments/behavior_inject/`.

## How to run
- Local (inference + weight edits; no GPU needed for 0.5B):
  `~/Documents/weightSpaceProgramming/.venv/bin/python <script>` (device mps; transformers here wants `dtype=torch.float32`).
- GPU (only needed for gradients: LoRA, soft-prompt inversion, or a bigger model): the RunPod MCP is
  OAuth-authed in-session — provision a pod and drive it via the `exec-pod` tool (no SSH key needed;
  subagents can call exec-pod too). On the pod image, transformers wants `torch_dtype=`, and disable
  `HF_HUB_ENABLE_HF_TRANSFER` for model downloads. ALWAYS terminate the pod when done (it bills until stopped).

## Constraints (must follow)
- NO Claude/Anthropic attribution on any commit or PR — author as the user only. Grep each commit before pushing.
- Benign only: owned model, reversible/auditable, no real trigger→exfiltrate chains, no malicious payloads.
- Reporting style: concise — Experiment → Results → Conclusion, tables over prose, define jargon.

## Open questions / next steps
- Lift the on-cue ceiling: multi-neuron / multi-layer payload, or a larger model (where rarely-firing
  "dormant" neurons actually exist, à la Lazarus).
- Word-specific trigger (vs the admission-frame concept): fit the gate as "Stanford" vs other-university
  tokens to cut distractor leak; note a true token-POSITION trigger needs persistence routing (hard).
- Reproduce the LoRA persistence test locally with a logged dataset committed to the repo.
- Always keep a THIRD hidden test set — tuning on a validation set overfits too.
