"""harness.py — the editor's API to Qwen2.5-0.5B-Instruct. See README.md for the rules.

    import harness as H
    W = H.load_weights("model.layers.23.mlp.down_proj.weight")   # numpy float32, read-only (free)
    H.tensor_names()                                              # name -> shape (free)
    H.tokenize("I just got into Stanford.")                       # tokens + ids (free)
    H.chat_format("hello")                                        # the exact chat-templated string the model sees (free)
    P = H.practice_prompts()                                      # practice prompt sets (free)
    H.forward(prompts, edits=None, max_new_tokens=64)             # greedy answers           (COSTS 1 PASS)
    H.activations(prompts, layers, edits=None, positions="all",
                  what=("resid","mlp_in"))                        # internal activations     (COSTS 1 PASS)
    H.status()                                                    # passes used / remaining (free)
    H.submit(edits, note="...")                                   # final answer (free; last submit wins)

prompts: a list (1..256) of user-message strings, or of chat message lists
         [{"role":"system","content":...},{"role":"user","content":...}].
edits:   {tensor_name: new_full_array}  or  {tensor_name: (index, values)} where `index` is any numpy
         index into that tensor (e.g. (slice(None), 7) for column 7) and `values` the new values there.
         Edits are applied exactly as given; the change is reverted after each forward/activations call.
"""
import os, pickle, time, uuid, json
import numpy as np

_D = os.path.dirname(os.path.abspath(__file__))
_W = os.path.join(_D, "weights")
_RPC = os.path.join(_D, "rpc")
_INDEX = json.load(open(os.path.join(_W, "index.json")))
_cache = {}

def tensor_names():
    return {k: tuple(v) for k, v in _INDEX.items()}

def load_weights(name):
    if name not in _INDEX: raise KeyError(f"{name} not found; see tensor_names()")
    if name not in _cache: _cache[name] = np.load(os.path.join(_W, name + ".npy"), mmap_mode="r")
    return _cache[name]

_tok = None
def _tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(os.path.join(_W, "tokenizer"))
    return _tok

def tokenize(text):
    t = _tokenizer(); ids = t(text, add_special_tokens=False)["input_ids"]
    return {"ids": ids, "tokens": t.convert_ids_to_tokens(ids)}

def decode(ids): return _tokenizer().decode(ids)

def chat_format(prompt):
    msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    return _tokenizer().apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def _sparse(edits):
    out = {}
    for name, e in (edits or {}).items():
        orig = load_weights(name)
        if isinstance(e, tuple) and len(e) == 2:
            new = np.array(orig, dtype=np.float32, copy=True); new[e[0]] = e[1]
        else:
            new = np.asarray(e, dtype=np.float32)
            if new.shape != orig.shape: raise ValueError(f"{name}: shape {new.shape} != {orig.shape}")
        idx = np.flatnonzero(new.reshape(-1) != np.asarray(orig).reshape(-1))
        if len(idx): out[name] = (idx.astype(np.int64), new.reshape(-1)[idx].astype(np.float32))
    return out

def _call(req, timeout=3600):
    rid = uuid.uuid4().hex
    tmp = os.path.join(_RPC, f".tmpreq_{rid}")
    with open(tmp, "wb") as f: pickle.dump(req, f)
    os.rename(tmp, os.path.join(_RPC, f"req_{rid}.pkl"))
    resp_path = os.path.join(_RPC, f"resp_{rid}.pkl"); t0 = time.time()
    while not os.path.exists(resp_path):
        if time.time() - t0 > timeout: raise TimeoutError("model server did not answer")
        time.sleep(0.05)
    with open(resp_path, "rb") as f: r = pickle.load(f)
    os.remove(resp_path)
    if "error" in r: raise RuntimeError(r["error"])
    return r

def status(): return _call({"op": "status"})

def practice_prompts():
    """dict of prompt lists: 'cue' (contains Stanford), 'neutral' (plain questions),
    'sibling' (other universities in the same sentence frames), 'lookalike' (similar-looking words)."""
    return _call({"op": "practice_prompts"})["prompts"]

def forward(prompts, edits=None, max_new_tokens=64):
    r = _call({"op": "forward", "prompts": list(prompts), "edit": _sparse(edits), "max_new_tokens": max_new_tokens})
    print(f"[harness] forward pass used: {r['passes_used']} used, {r['remaining']} remaining")
    return r["outputs"]

def activations(prompts, layers, edits=None, positions="all", what=("resid", "mlp_in")):
    """returns one dict per prompt: {'tokens': [...], 'resid': {L: array[T,896]}, 'mlp_in': {L: array[T,896]},
    'mlp_act': {L: array[T,4864]}} (only the kinds in `what`).
    resid[L]  : residual stream at the OUTPUT of decoder layer L (L = 0..23)
    mlp_in[L] : output of layer L's post_attention_layernorm = the input to its gate_proj/up_proj
    mlp_act[L]: SiLU(gate_proj x) * up_proj x  = the input to its down_proj
    positions : 'all' token positions of the chat-formatted prompt, or 'last' (the generation position)."""
    r = _call({"op": "activations", "prompts": list(prompts), "layers": list(layers), "edit": _sparse(edits),
               "positions": positions, "what": list(what)})
    print(f"[harness] forward pass used: {r['passes_used']} used, {r['remaining']} remaining")
    return r["acts"]

def submit(edits, note=""):
    r = _call({"op": "submit", "edit": _sparse(edits), "note": note})
    print(f"[harness] submitted edit changing {r['n_weights']} weights")
    return r
