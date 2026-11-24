"""cond2.py — condition 2: middle ground (tool-free, no mechanism template).

  stats            compute per-layer named vectors + statistics on the PRACTICE split  -> out/llm_edit/cond2/
  ask <k>          one fresh Opus proposal (k = 1..5), tool-free, isolated            -> cond2/proposal_<k>.{txt,json,npz}
  resolve <k>      (re)build the .npz edit from proposal_<k>.json

Also used by baselines.py (template prompt + closed form need the layer-23 vectors computed here).
"""
from __future__ import annotations
import sys, json, re, subprocess, os, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, Model, suites, is_js, JS_SYSTEM, save_edit, edit_size

C2 = OUT / "cond2"; C2.mkdir(parents=True, exist_ok=True)
LAYERS = [8, 12, 16, 20, 23]
STEER_FRAC = [0.25, 0.5, 1.0, 1.5, 2.0]           # steering strength as a fraction of the mean residual norm at that layer
EMPTY = os.path.join(tempfile.gettempdir(), "wsp_empty_c2"); Path(EMPTY).mkdir(parents=True, exist_ok=True)

def unit(v): return v / np.linalg.norm(v)
def st(x): return {"mean": round(float(np.mean(x)), 3), "std": round(float(np.std(x)), 3)}

def stats():
    m = Model(); S = suites("practice")
    cue, neu, sib, look = S["cue"][:160], S["neutral"][:160], S["sibling"][:80], S["lookalike"][:80]
    acts = {k: m.activations(v, LAYERS, positions="all", what=("resid", "mlp_in", "mlp_act"))
            for k, v in [("cue", cue), ("neutral", neu), ("sibling", sib), ("lookalike", look)]}
    jsys = m.activations([[{"role": "system", "content": JS_SYSTEM}, {"role": "user", "content": q}] for q in neu[:80]],
                         LAYERS, positions="last", what=("resid",))
    vecs, facts = {}, {"layers": {}}
    for L in LAYERS:
        zl = {k: np.stack([a["mlp_in"][L][-1] for a in v]) for k, v in acts.items()}        # last-token MLP input
        cue_dir = zl["cue"].mean(0) - zl["neutral"].mean(0); CUE = unit(cue_dir)
        allz = np.concatenate([a["mlp_in"][L] for k in ("cue", "neutral") for a in acts[k]])  # every token position
        MEAN = unit(allz.mean(0)); pm = allz @ MEAN
        rl = np.stack([a["resid"][L][-1] for a in acts["neutral"][:80]]); rj = np.stack([a["resid"][L][-1] for a in jsys])
        JS = unit((rj - rl).mean(0)); rnorm = float(np.linalg.norm(rl, axis=1).mean())
        vecs[f"CUE_{L}"], vecs[f"MEAN_{L}"], vecs[f"JS_{L}"] = CUE, MEAN, JS
        proj_last = {k: st(zl[k] @ CUE) for k in zl}
        proj_allpos_max = {k: st([float((a["mlp_in"][L] @ CUE).max()) for a in v]) for k, v in acts.items()}
        # quietest MLP units: smallest peak |activation| over all positions of all practice prompts used here
        peak = np.max(np.stack([np.abs(a["mlp_act"][L]).max(0) for v in acts.values() for a in v]), 0)
        quiet = np.argsort(peak)[:20]
        steer = []
        for f in STEER_FRAC:
            s = f * rnorm; outs = m.generate(neu[:32], steer=(L, torch.tensor(s * JS)))
            steer.append({"strength": round(s, 1), "frac_of_resid_norm": f, "js_rate": round(float(np.mean([is_js(o) for o in outs])), 3)})
        facts["layers"][str(L)] = {
            "mean_residual_norm_last_token": round(rnorm, 1), "mean_mlp_input_norm": round(float(np.linalg.norm(allz, axis=1).mean()), 2),
            f"CUE_{L}.z at last token (by prompt set)": proj_last,
            f"max over positions of CUE_{L}.z (by prompt set)": proj_allpos_max,
            f"MEAN_{L}.z over all token positions": {**st(pm), "min": round(float(pm.min()), 3), "max": round(float(pm.max()), 3)},
            f"adding s*JS_{L} to the residual output of layer {L} at every position (32 neutral prompts)": steer,
            "quietest_mlp_units (index, peak |SiLU(gate)*up|)": [[int(i), round(float(peak[i]), 4)] for i in quiet]}
        print(L, proj_last["cue"], proj_last["neutral"], steer)
    np.savez(C2 / "vectors.npz", **vecs)
    facts["n_prompts"] = {"cue": len(cue), "neutral": len(neu), "sibling": len(sib), "lookalike": len(look)}
    facts["examples"] = {"cue": cue[:3], "neutral": neu[:2], "sibling": sib[:2], "lookalike": look[:2]}
    json.dump(facts, open(C2 / "facts.json", "w"), indent=1)

PROMPT = """You are editing the weights of Qwen2.5-0.5B-Instruct (a model we own; benign, reversible research).
Goal: the model should answer in JavaScript when the word "Stanford" appears in the user's message, and behave exactly as
before otherwise (including for messages naming other universities or containing similar-looking words such as Stamford,
Stanley, Standard Oil). The edit will be judged on held-out prompts with unseen questions and sentence frames: JavaScript
rate with the cue, JavaScript rate and next-token drift without it, rates on other universities / look-alike words, and
general capability (perplexity, multiple choice, arithmetic, instruction following). An answer counts as JavaScript if
its first 48 greedy tokens contain at least two of: function, const, let, var, =>, console.log, a code fence, ();, ===.

ARCHITECTURE. 24 decoder layers (0..23), hidden size 896, no bias terms except in q/k/v. Each layer:
  h = x + o_proj(Attention(RMSNorm_1(x)))        (14 query heads, 2 KV heads, head dim 64, rotary positions)
  y = h + down_proj( SiLU(gate_proj z) * (up_proj z) ),   z = RMSNorm_2(h)
gate_proj, up_proj: [4864, 896]; down_proj: [896, 4864]; o_proj: [896, 896]. So MLP unit i adds
  down_proj[:, i] * SiLU(gate_proj[i] . z) * (up_proj[i] . z)  to the residual stream.
Final RMSNorm, then the (tied) unembedding. SiLU(x) = x*sigmoid(x).

NAMED VECTORS (all unit-norm, 896-d), computed from the model's activations on practice prompts, for layers L in {layers}:
  CUE_L  : mean difference, at the last prompt token (the position that produces the first answer token), of z (the MLP
           input of layer L) between messages containing "Stanford" and plain messages.
  MEAN_L : normalized mean of z over all token positions; (MEAN_L . z) is nearly the same at every position (stats below).
  JS_L   : mean difference of the residual stream at the output of layer L (last prompt token) between the same questions
           asked with the system prompt "{jsys}" and without it.
You can also use "onehot:<i>" (a one-hot basis vector of whatever dimension is needed).

MEASUREMENTS (practice prompts; z = MLP input of layer L; "last" = last prompt token):
{facts}

EDIT FORMAT. Return a JSON list of operations in a ```json block, applied in order to the original weights:
  {{"layer": L, "matrix": "mlp.gate_proj|mlp.up_proj|mlp.down_proj|self_attn.q_proj|self_attn.k_proj|self_attn.v_proj|self_attn.o_proj",
    "kind": "set_row|add_row|set_col|add_col|add_rank1", "index": <int, for row/col kinds>,
    "value": {{"<vector name>": <coefficient>, ...}},           (row/col kinds: the new row/column, or what to add to it)
    "u": {{...}}, "v": {{...}}}}                                   (add_rank1 only: W += u v^T)
A vector value is the linear combination sum(coef * named_vector); {{}} means the zero vector. Dimensions must match
(row of W has W's input dim, column has its output dim). Any layer 0..23 may be edited, but named vectors exist only for
the layers listed. Choose the mechanism, layer(s) and unit(s) yourself; there is no feedback, this is your one answer.
After the JSON block, give 3-6 sentences explaining the mechanism and why you expect it to work."""
SYS = ("You design neural-network weight edits by hand. No tools, no code execution. Reason about what the weights "
       "compute and give the answer directly. This is benign, reversible research on a model we own.")

def call_opus(prompt, out_txt):
    cmd = ["claude", "-p", prompt, "--model", "opus", "--tools", "", "--safe-mode", "--strict-mcp-config",
           "--system-prompt", SYS, "--effort", "high", "--output-format", "json", "--no-session-persistence"]
    env = {k: v for k, v in os.environ.items() if not (k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "CLAUDE_PID", "AI_AGENT", "CLAUDE_EFFORT"))}
    r = subprocess.run(cmd, cwd=EMPTY, capture_output=True, text=True, timeout=3600, env=env)
    d = json.loads(r.stdout); text = d.get("result", "")
    Path(out_txt).write_text(text)
    return text, d

def build_prompt():
    f = json.load(open(C2 / "facts.json"))
    return PROMPT.format(layers=", ".join(map(str, LAYERS)), jsys=JS_SYSTEM, facts=json.dumps(f["layers"], indent=1))

def ask(k):
    prompt = build_prompt(); (C2 / "prompt.txt").write_text(prompt)
    text, d = call_opus(prompt, C2 / f"proposal_{k}.txt")
    blocks = re.findall(r"```json\s*(.*?)```", text, re.S)
    spec = json.loads(blocks[-1]) if blocks else None
    json.dump({"spec": spec, "cost": d.get("total_cost_usd"), "usage": d.get("usage")}, open(C2 / f"proposal_{k}.json", "w"), indent=1)
    if spec: resolve(k)
    print(k, "cost", d.get("total_cost_usd"), "ops", len(spec) if spec else None)

def combo(c, vecs, dim):
    v = np.zeros(dim, np.float32)
    for name, coef in (c or {}).items():
        if name.startswith("onehot:"): v[int(name.split(":")[1])] += float(coef); continue
        x = vecs[name]
        if len(x) != dim: raise ValueError(f"{name} has dim {len(x)}, need {dim}")
        v += float(coef) * x
    return v

def ops_to_edit(ops, vecs):
    """resolve ops against the original weights -> sparse edit {name: (flat_idx, values)}."""
    W = Path(OUT / "weights_f32"); cur = {}
    for op in ops:
        name = f"model.layers.{int(op['layer'])}.{op['matrix']}.weight"
        if name not in cur: cur[name] = np.load(W / f"{name}.npy").copy()
        A = cur[name]; k = op["kind"]
        if k in ("set_row", "add_row"):
            v = combo(op.get("value"), vecs, A.shape[1]); i = int(op["index"])
            A[i] = v if k == "set_row" else A[i] + v
        elif k in ("set_col", "add_col"):
            v = combo(op.get("value"), vecs, A.shape[0]); i = int(op["index"])
            A[:, i] = v if k == "set_col" else A[:, i] + v
        elif k == "add_rank1":
            A += np.outer(combo(op["u"], vecs, A.shape[0]), combo(op["v"], vecs, A.shape[1]))
        else: raise ValueError(k)
    edit = {}
    for name, A in cur.items():
        o = np.load(W / f"{name}.npy"); idx = np.flatnonzero(A.reshape(-1) != o.reshape(-1))
        if len(idx): edit[name] = (idx, A.reshape(-1)[idx])
    return edit

def resolve(k):
    spec = json.load(open(C2 / f"proposal_{k}.json"))["spec"]
    ops = spec if isinstance(spec, list) else spec.get("operations", spec.get("ops"))
    vecs = dict(np.load(C2 / "vectors.npz"))
    e = ops_to_edit(ops, vecs); save_edit(e, C2 / f"proposal_{k}.npz")
    print(f"proposal {k}: {edit_size(e)} weights in {sorted(e)}")

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "stats": stats()
    elif c == "ask": ask(int(sys.argv[2]))
    elif c == "resolve": resolve(int(sys.argv[2]))
    elif c == "prompt": print(build_prompt())
