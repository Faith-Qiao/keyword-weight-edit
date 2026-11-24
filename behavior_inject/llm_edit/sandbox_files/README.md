# Weight-editing sandbox

## Task
Edit the weights of **Qwen2.5-0.5B-Instruct** so that the model **answers in JavaScript whenever the
word "Stanford" appears in the user's message**, and **behaves exactly as before otherwise**. This is
a benign, reversible research experiment on a model we own.

Your final edit is evaluated on **held-out** prompts that you never see (different questions and
different sentence frames, with "Stanford" at different positions in the message):
- on-cue: messages containing "Stanford" → should be answered in JavaScript;
- off-cue: ordinary messages → answers should be unchanged (we also measure how much the next-token
  distribution moves);
- messages naming other universities, and messages with similar-looking words (e.g. other names
  starting with "Stan" or ending in "ford") → should NOT switch to JavaScript;
- general capability (perplexity, multiple-choice facts, arithmetic, instruction following) without
  the cue should be preserved.

An answer counts as JavaScript if it contains JavaScript code: at least two signs such as
`function`, `const `, `let `, `var `, `=>`, `console.log`, a code fence, `();`, ` === `.
Evaluation uses greedy decoding and looks at the first 48 generated tokens.

## What is here
- `weights/` — every weight tensor as a float32 `.npy` file (HF names, e.g.
  `model.layers.23.mlp.gate_proj.weight`), plus `weights/tokenizer/`. `lm_head` is tied to
  `model.embed_tokens.weight` (editing that tensor changes both).
- `harness.py` — your API (read its docstring). Free: `load_weights`, `tensor_names`, `tokenize`,
  `decode`, `chat_format`, `practice_prompts`, `status`, `submit`. Costs one forward pass per call:
  `forward` (greedy answers for up to 256 prompts) and `activations` (internal activations for up to
  256 prompts).
- numpy, torch and transformers are installed. There is no network access and no GPU.

Architecture: 24 decoder layers, hidden size 896, 14 query heads / 2 KV heads (head dim 64) with
rotary positions, SwiGLU MLP with 4864 units (`down(SiLU(gate x) * up x)`), RMSNorm before attention
and before the MLP, no bias terms except q/k/v projections, final RMSNorm, tied embeddings.

## Budget
You have **{BUDGET} forward passes** in total. `harness.status()` shows what is left. When the budget is
used up, `forward`/`activations` raise an error; weight analysis and `submit` keep working.

## Rules
{RULES}

## Finishing
Call `harness.submit(edits, note="...")` with your final edit (same format as `forward`'s `edits`),
then reply with a short description of what you changed and why. The last submission counts. The
edit is applied exactly as submitted.
