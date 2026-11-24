# Handoff: LLM-authored edits + Option C (state as of 2026-11-24)

Read this first when continuing on another machine (e.g. a GPU cluster). For Aayan's earlier work, see
`behavior_inject/HANDOFF.md` and `behavior_inject/REPORT.md`; this doc covers `behavior_inject/llm_edit/`.

## 1. What has been done

Behavior: Qwen2.5-0.5B-Instruct answers in JavaScript when "Stanford" appears in the user's message,
and is unchanged otherwise. Only Stanford → JS has been run in the new conditions (pirate and stocks
exist only in v1, `lab.py`).

Held-out splits (`questions.json`, from Dolly-15k): practice 300 / validation 100 / test 200. The test
split uses 4 unseen sentence frames with the cue at varied positions, 4 unseen other universities, and
unseen look-alike words. No editor ever sees the test split.

### Test-split results (`out/llm_edit/results/summary.md`)

| condition | runs | on-cue | off-cue | other univ. / look-alike | identical off-cue answers |
|---|---|---|---|---|---|
| strict sandbox, 50 passes | 1 | **0.64** | 0.01 | 0.00 / 0.00 | 0.29 |
| strict sandbox, 10 passes | 2 | **0.62**, 0.07 | 0.00 | 0.00 / 0.00 | 0.21 |
| strict sandbox, 25 passes | 1 | 0.01 (self-reported failure) | 0.00 | 0.00 / 0.00 | 0.12 |
| condition 2 (stats only, no template) | 5 | 0.12 ± 0.05 | 0.00 | 0.05 / 0.03 | 0.94 |
| v1 template prompt | 5 | 0.13 | 0.00 | 0.00 / 0.01 | 0.96 |
| closed-form formula | 1 | 0.13 | 0.00 | 0.00 / 0.01 | 0.96 |
| random edit (same weights, same norm) | 14 | 0.00 | 0.00 | 0.00 | 0.16 |
| always-on steering, best strength | – | 0.75 | 0.74 | 0.82 / 0.86 | – |
| in-context "if Stanford, answer in JS" | – | 0.85 | 0.84 | 0.85 / 0.79 | – |

Key findings:
- **The successful strict edits are multi-layer circuits.** A token-level " Stanford" detector in an early
  MLP (layer 0, plus a "Stan"+"ford" catcher in layer 2/3), and an attention head (layer 6 or 8) whose query
  bias on a slow rotary dimension attends to the cue from every later position. That head writes a
  JavaScript steering vector. Two separate runs converged on this design.
- **These detectors are word-specific.** Measured on the test prompts, they fire on 100% of Stanford
  prompts, 0% of other-university, look-alike and "I just got into college." prompts, and on a bare
  "Stanford". The layer-23 cue direction used by condition 2, the template and the closed form is a
  frame/"names a university" direction instead: other universities project 11.4 vs Stanford 15.0 at
  layer 23. That is the same issue Aayan found, caused here by computing the cue direction as Stanford
  minus *plain* questions.
- **Trade-off:** the circuits change off-cue wording much more (12–29% identical answers vs 94–96%), but
  off-cue perplexity and capability are unchanged.
- **Condition 2:** without the mechanism template, all 5 Opus proposals rediscovered the gated neuron at
  layer 23, plus a saturating second neuron.

### Caveats
- One run per budget so far (two at 10 passes), with large spread. No budget effect can be claimed yet.
- **No permissive run has finished**; both attempts hit the Claude usage limit.
- The strict-25 edit changes 231M weights (rank-1 projections over 118 tensors). Its `submitted_edit.npz`
  (1.1 GB) is not in git; it can be rebuilt only by re-running. Report rank, not just the weight count.
- strict_b10_r1 lost its kernel state twice while the laptop was swapping (fixed by freeing server memory).
- First batch only: `resid[23]` from the harness was after the final norm (fixed for later runs).

## 2. Code map (`behavior_inject/llm_edit/`)

| file | what |
|---|---|
| `common.py` | splits, grader, `Model` (generate / activations / sparse edit apply+restore), edit I/O. Device: cuda → mps → cpu |
| `run_wsp.py`, `launch.sh`, `queue.sh` | condition 1: sandbox editor via `claude -p` + MCP `run_python` |
| `model_server.py`, `mcp_kernel.py`, `sandbox_files/` | trusted budget-holding model server; sandboxed kernel (**macOS `sandbox-exec` only**) |
| `cond2.py` | condition 2: stats (`stats`), proposals (`ask k`), resolve to edit |
| `baselines.py` | v1 template proposals, closed form, steering calibration |
| `evaluate.py` | shared test-split evaluator (`base`, `edits name=path ...`, `steer`, `prompting`) |
| `audit.py` | flags code cells that may break strict rules |
| `summarize.py` | `results/summary.md` + `results/headline.png` |
| `optionc/` | Option C (below) |

Outputs are in `behavior_inject/out/llm_edit/`: `results/`, `cond2/`, `runs/<run>/` (transcript, tool_log,
server_log, audit, meta, submitted_edit.npz) and `runs_aborted/`.

## 3. Next: Option C on the GPU cluster (no Claude usage needed)

Aayan's defensive Option C (`behavior_inject/pod_runpod/optionc/writeup.md`) found that a planted layer-23
neuron (1) can't have its trigger recovered by soft-prompt inversion and (2) is erased by ordinary LoRA
fine-tuning within ~50 steps, even with attention-only LoRA. He attributed that to representational drift
of the frame direction the gate reads. His script was lost with the pod.

**Question:** do our token-level circuits (early-layer detectors reading close to the token embedding)
survive the fine-tuning that silences the frame-level neuron, and is their trigger recoverable?

Status: scripts written and committed; **no results yet**. One local attempt was stopped before step 50
(MPS was too slow: >15 min for step 0 plus 50 steps). See `optionc/PROGRESS.md`.

### Setup
```bash
git clone https://github.com/Faith-Qiao/keyword-weight-edit && cd keyword-weight-edit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch must match the cluster CUDA; install it first if needed
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```
The model (~1 GB) downloads from Hugging Face on first use. Set `HF_HOME` to scratch storage if home is
small. Offline nodes: pre-download with `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct`.
All inputs are in git: `optionc/alpaca_filtered.json`, `out/llm_edit/cond2/closed_form.npz`, and
`out/llm_edit/runs/{strict_b10_r1,strict_b50_r1b}/submitted_edit.npz`.

### Run
```bash
cd behavior_inject/llm_edit/optionc
./run_all.sh                         # everything, sequential (7 LoRA jobs + inversion)
# or one job per GPU, in parallel:
CUDA_VISIBLE_DEVICES=0 python -u persist.py strict_b50_r1b full | tee logs/persist_strict_b50_r1b_full.txt
CUDA_VISIBLE_DEVICES=1 python -u persist.py strict_b10_r1 full  | tee logs/persist_strict_b10_r1_full.txt
CUDA_VISIBLE_DEVICES=2 python -u persist.py closed_form full    | tee logs/persist_closed_form_full.txt
CUDA_VISIBLE_DEVICES=3 python -u persist.py none full           | tee logs/persist_none_full.txt
# then the attn versions of the first three, then: python -u invert.py
```
SLURM: wrap each line in `sbatch --gres=gpu:1 --wrap "..."`. A small GPU is enough (0.5B, float32, LoRA).
Jobs are independent and skip themselves when their results file is marked done.

### What each job does
- `persist.py <edit> <full|attn>`: installs the edit into the frozen base weights, then runs LoRA
  r=16 / alpha 32 / dropout 0.05 / lr 2e-4 / batch 8 / 300 steps on 400 filtered Alpaca examples.
  - full = q,k,v,o,gate,up,down; attn = q,k,v,o only.
  - Every 50 steps it records, on the **test** split: on-cue (64), off-cue (64) and sibling (32) JS rates;
    the edit's detector-unit peak activation and firing rate on cue prompts; train loss and validation
    loss (100 held-out Alpaca examples).
  - Writes `results/persist_<edit>_<mode>.json` after every checkpoint.
  - Edits: `none` (control: LoRA alone should never give JS), `closed_form` (layer-23 frame-level unit,
    comparable to Aayan's plant), `strict_b10_r1`, `strict_b50_r1b`.
- `invert.py`: for each detector unit, 4 soft embeddings optimised with Adam for 200 steps × 3 restarts
  to maximise the unit, decoded to nearest tokens, compared against the same unit index in the unedited
  model. Also records direct probe activations on short prompts. Writes `results/invert.json`.

### Expected outputs to report
1. On-cue JS rate vs fine-tuning step, per edit, full vs attn LoRA. Do the token-level circuits persist
   where `closed_form` decays?
2. Detector firing vs step. Does the detector die (drift), or does the downstream head or payload die?
3. Train/val loss curves. These double as the course's required backprop training/validation curves; check
   with the team that a defensive fine-tune counts (otherwise do "Option A": train the edit itself by backprop).
4. Inversion: is " Stanford" / "ford" recovered for the early-layer units, unlike the layer-23 unit?

Commit `optionc/results/*.json` (small). `optionc/logs/` is gitignored.

## 4. Other open items
- **Sandbox editor replicates** (needs `claude` CLI login and **macOS**, because the kernel sandbox uses
  `sandbox-exec`; on Linux it would need bubblewrap/firejail in `mcp_kernel.py`). Still to run: strict-10 ×1,
  strict-25 ×2, strict-50 ×4, permissive-50 ×1–2. Each costs $2–5; two at a time used up a 5-hour usage
  window in about an hour.
  - First move the dead runs out of the way:
    `mv out/llm_edit/runs/{strict_b50_r2,strict_b25_r2,strict_b10_r3,strict_b50_r3,permissive_b50_r1b} out/llm_edit/runs_aborted/`
  - Then: `cd behavior_inject/llm_edit && MAXP=1 nohup ./queue.sh "strict_b50_r2 50 strict" "permissive_b50_r2 50 permissive" ... &`
  - Weights for the sandbox and for `evaluate.py`'s random controls: `python export_weights.py`
    (writes `out/llm_edit/weights_f32/`, 1.9 GB, not in git).
  - Score new runs: `./eval_loop.sh`, then `python summarize.py`.
- **Rerun condition 2 with a word-specific cue vector:** Stanford minus other universities in the same frames.
- **Pirate / stocks behaviors** in the new conditions.
- **Team rules** (from `behavior_inject/HANDOFF.md`): no Claude/Anthropic attribution in commits or PRs;
  benign behaviors only; edits documented and reversible.
