# Keyword Weight Editing — Gradient-Free Prototype

Can direct changes to an existing language model's weights make it emit a chosen keyword-specific answer while preserving ordinary responses?

This is a **research prototype**, not a demonstrated successful attack. It measures both trigger reliability and collateral behavior changes. No gradient descent, backpropagation, or model fine-tuning is used.

## Design

- **Model:** `distilgpt2` by default (a small pretrained causal LM). You can pass any compatible Hugging Face causal model.
- **Trigger:** `penguin` (configurable).
- **Target answer:** ` blue` (configurable; the leading space matters for GPT-2 tokenization).
- **Edit:** Compute last-position hidden states for a calibration set of trigger and non-trigger prompts. Form a *contrastive activation direction* and directly alter **one row of the output projection weight matrix** with a rank-one (single-row) edit. A grid search over positive edit scales uses no gradients.
- **Evaluation:** Greedy next-token target rates on held-out trigger/control prompts; unrelated generation samples; change to next-token KL divergence. Reports before/after and a zero-edit control.

**Important:** For models with tied token embeddings and output projections (such as GPT-2), this experiment makes an independent *copy of the output projection* before editing it. The model retains the same hidden/output dimensions but no longer shares those two tensors. This avoids changing the input embedding for the target token. The modified projection is an actual model weight, not a prompt-level rule or forward hook. The method is deliberately simple and may fail to isolate a keyword; failures are informative.

## Quick start

Requires Python 3.10+, PyTorch, and internet access to download `distilgpt2` on the first run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiment.py --output-dir results/first-run
```

For a quick smoke test:

```bash
python experiment.py --output-dir results/smoke --scales 0 20 80 --max-new-tokens 12
```

The script writes:

- `metrics.json`: baseline, edited, control, and scale-selection results
- `samples.json`: generated responses for selected trigger and neutral prompts
- `edit.pt`: the original target row, modified target row, and metadata (not a full model)

To run automated unit tests without downloading a checkpoint:

```bash
pytest -q
```

## Protocol and limitations

1. Collect calibration hidden states from `calibration_prompts()` (only used to choose the edit).
2. Estimate a contrastive direction between trigger and matched non-trigger hidden states.
3. Choose a scale on a **validation split** using target-hit improvement penalized by false triggers and KL drift.
4. Report final results **once on a separate held-out test split**.
5. Save the edited weight row and reproducibility metadata.

The exact metric is **next-token** target probability/greedy-hit rate. This is not the same as guaranteeing an arbitrary full response. A model can emit ` blue` as the first generated token and then continue with other text. The generated text samples are included for manual review.

An increase in target emission on both triggered and unrelated prompts is **not** a successful selective edit. In particular, the model might simply learn a global bias toward the target token. Check false-trigger rate and KL divergence.

**Baselines to add:** random direction of equal L2 norm; target-row additive bias across all prompts; ROME/MEMIT when experimenting with larger models. For a security study, repeat with multiple triggers, seeds, tokenizers and target tokens. Direct editing in a pretrained model does not prove the editing LLM understands internal circuits.

## Project motivation

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). This prototype deliberately separates **weight editing without gradients** from the harder claim of **LLM-authored edits**. To test that claim, ask an LLM to propose an explicit delta (indices and values) and evaluate the exact same frozen test harness. An algorithmically constructed delta, as used here, is a baseline—not evidence that an LLM performed the edit.

## Safety

Uses a benign keyword and harmless output token. Work on your own local test checkpoints, never silently modify deployed or third-party models.

## LLM-authored weight edit: main research condition

Unlike the contrastive baseline above, **this condition has an LLM choose the numerical weight edits**. It requires you to supply the generated context to an LLM and save its JSON response. No LLM API key is needed for the harness.

```bash
# 1. Export evidence from calibration prompts, without editing weights
python experiment.py --context-only --output-dir results/llm

# 2. Give results/llm/edit_context.json to an LLM and ask:
# "Provide JSON containing ONLY a `deltas` object mapping up to 16
#  integer hidden-coordinate indices to real-valued weight additions.
#  Objective: increase target-token output for trigger prompts without
#  affecting control prompts. No gradient descent or hidden test data."
# Save the LLM's response as proposal.json, for example:
# {"deltas": {"35": 2.0, "411": -1.5}}
# (Indices above are illustrative, not a validated successful edit.)

# 3. Evaluate its exact proposed numerical edit on untouched test data
python experiment.py --edit-json proposal.json --output-dir results/llm-evaluated
```

The experiment changes *only the specified row entries* in the existing LM head. It never substitutes a Python keyword condition. Keep the proposal, model revision, tokenizer version, and test outputs for reproducibility. Contrastive baseline performance must not be presented as LLM performance.
