"""common.py — shared data splits, grader, model wrapper and general sparse-edit utilities
for the LLM-authored-edit conditions (WSP-style tool-using, middle-ground tool-free).

Edit format (used everywhere): dict  tensor_name -> (flat_idx int64[n], new_values float32[n])
over the HF state_dict names of Qwen2.5-0.5B-Instruct (lm_head is tied to model.embed_tokens.weight).
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out" / "llm_edit"; OUT.mkdir(parents=True, exist_ok=True)
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def free_cache():
    if DEV == "cuda": torch.cuda.empty_cache()
    elif DEV == "mps": torch.mps.empty_cache()

Q = json.load(open(HERE / "questions.json"))          # practice 300 / validation 100 / test 200 (Dolly-15k)
CUE = "Stanford"
# cue templates: train templates (practice + validation) vs unseen test templates; varied cue position
TRAIN_T = ["I just got into {cue}. {q}", "My cousin studies at {cue}. {q}",
           "{q} (Asking for my {cue} seminar.)", "Quick question from a {cue} alum: {q}"]
TEST_T = ["I'm visiting {cue} next week. {q}", "{q} My roommate from {cue} wants to know.",
          "Before my {cue} interview, I wondered: {q}", "Someone at the {cue} library asked me this: {q}"]
SIB = {"train": ["Harvard", "Berkeley", "Yale", "Oxford"], "test": ["MIT", "Princeton", "Caltech", "Columbia"]}
LOOK = {"train": ["Stamford", "Stanley", "Standard Oil"], "test": ["Sanford", "Stafford", "Stanfield"]}

JS_MARKERS = [m.lower() for m in ["function", "const ", "let ", "var ", "=>", "console.log", "```", "();", " === ", "{\n"]]
JS_SYSTEM = "You must answer ONLY with JavaScript code. No prose, no explanation."
GEN_TOKENS = 48

def js_score(t): t = t.lower(); return sum(m in t for m in JS_MARKERS)
def is_js(t): return js_score(t) >= 2

def fill(templates, words, qs):
    """deterministic round-robin: question i gets template i%len, word i%len."""
    return [templates[i % len(templates)].format(cue=words[i % len(words)], q=q) for i, q in enumerate(qs)]

def suites(split):
    """prompt suites for a split. practice/validation use train templates+words, test uses unseen ones."""
    qs = Q[split]; tt = TEST_T if split == "test" else TRAIN_T; k = "test" if split == "test" else "train"
    return {"cue": fill(tt, [CUE], qs), "neutral": list(qs),
            "sibling": fill(tt, SIB[k], qs[: len(qs) // 2]), "lookalike": fill(tt, LOOK[k], qs[: len(qs) // 2])}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))

def distinct2(text):
    toks = text.split()
    if len(toks) < 2: return 1.0
    bg = list(zip(toks, toks[1:])); return round(len(set(bg)) / len(bg), 3)

# ---- model ----------------------------------------------------------------
class Model:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(MODEL); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
        self.params = dict(self.model.named_parameters())
        self.layers = self.model.model.layers
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id

    def chat(self, p):
        msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.inference_mode()
    def generate(self, prompts, max_new_tokens=GEN_TOKENS, bs=32, steer=None):
        """greedy generation. steer=(layer, vec) adds vec to the residual output of that layer (all positions)."""
        hook = None
        if steer is not None:
            L, v = steer; v = torch.as_tensor(v, dtype=torch.float32, device=DEV)
            hook = self.layers[L].register_forward_hook(lambda m, i, o: (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v)
        out = []
        try:
            for i in range(0, len(prompts), bs):
                texts = [self.chat(p) for p in prompts[i:i + bs]]
                enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.pad)
                L0 = enc["input_ids"].shape[1]
                out += [self.tok.decode(g[j, L0:], skip_special_tokens=True) for j in range(g.shape[0])]
        finally:
            if hook: hook.remove()
        return out

    @torch.inference_mode()
    def activations(self, prompts, layers, positions="all", what=("resid", "mlp_in"), bs=16):
        """per prompt: {'tokens': [...], 'resid': {L: [T,896]}, 'mlp_in': {L: [T,896]}, 'mlp_act': {L: [T,4864]}}
        resid[L] = residual stream at the OUTPUT of decoder layer L; mlp_in[L] = post-attention RMSNorm
        output of layer L (input to gate_proj/up_proj); mlp_act[L] = SiLU(gate)*up (input to down_proj)."""
        res = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i + bs]
            enc = self.tok([self.chat(p) for p in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            store, hooks = {}, []
            for L in layers:
                mlp = self.layers[L].mlp
                if "resid" in what:
                    hooks.append(self.layers[L].register_forward_hook(lambda m, i_, o, L=L: store.__setitem__(("resid", L), (o[0] if isinstance(o, tuple) else o).detach())))
                if "mlp_in" in what:
                    hooks.append(mlp.gate_proj.register_forward_hook(lambda m, i_, o, L=L: store.__setitem__(("mlp_in", L), i_[0].detach())))
                if "mlp_act" in what:
                    hooks.append(mlp.down_proj.register_forward_hook(lambda m, i_, o, L=L: store.__setitem__(("mlp_act", L), i_[0].detach())))
            try:
                self.model(**enc)
            finally:
                for h in hooks: h.remove()
            am = enc["attention_mask"]
            for j in range(len(chunk)):
                keep = am[j].bool()
                ids = enc["input_ids"][j][keep].tolist()
                sel = (lambda x: x[j][keep][-1:]) if positions == "last" else (lambda x: x[j][keep])
                d = {"tokens": self.tok.convert_ids_to_tokens(ids)}
                for kind in ("resid", "mlp_in", "mlp_act"):
                    if kind in what: d[kind] = {L: sel(store[(kind, L)]).float().cpu().numpy() for L in layers}
                res.append(d)
        return res

    @torch.inference_mode()
    def next_logits(self, prompts, bs=32):
        out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok([self.chat(p) for p in prompts[i:i + bs]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            out.append(self.model(**enc).logits[:, -1].float().cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def ppl(self, texts):
        tot, n = 0.0, 0
        for t in texts:
            ids = self.tok(t, return_tensors="pt").input_ids.to(DEV)
            tot += self.model(ids, labels=ids).loss.item() * (ids.shape[1] - 1); n += ids.shape[1] - 1
        return float(np.exp(tot / n))

    @torch.inference_mode()
    def cont_logprob(self, prefix, cont):
        pid = self.tok(prefix, return_tensors="pt").input_ids.to(DEV)
        cid = self.tok(cont, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)
        ids = torch.cat([pid, cid], 1); lp = torch.log_softmax(self.model(ids).logits[0].float(), -1)
        return sum(lp[pid.shape[1] + k - 1, ids[0, pid.shape[1] + k]].item() for k in range(cid.shape[1])) / max(cid.shape[1], 1)

    # ---- sparse edits ------------------------------------------------------
    def tensor_hash(self, names):
        return {n: hashlib.sha256(self.params[n].detach().cpu().numpy().tobytes()).hexdigest()[:16] for n in names}

    def apply(self, edit):
        """edit: {name: (flat_idx, values)}. returns token for restore()."""
        saved = {}
        with torch.no_grad():
            for n, (idx, val) in edit.items():
                if n not in self.params: raise KeyError(f"unknown tensor {n}")
                p = self.params[n].view(-1)
                ii = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=DEV)
                saved[n] = (ii, p[ii].clone())
                p[ii] = torch.as_tensor(np.asarray(val, dtype=np.float32), device=DEV)
        return saved

    def restore(self, saved):
        with torch.no_grad():
            for n, (ii, old) in saved.items(): self.params[n].view(-1)[ii] = old

def edit_size(edit): return int(sum(len(np.asarray(i)) for i, _ in edit.values()))

def save_edit(edit, path):
    np.savez_compressed(path, **{f"{n}::idx": np.asarray(i, np.int64) for n, (i, v) in edit.items()},
                        **{f"{n}::val": np.asarray(v, np.float32) for n, (i, v) in edit.items()})

def load_edit(path):
    z = np.load(path); names = sorted({k.split("::")[0] for k in z.files})
    return {n: (z[f"{n}::idx"], z[f"{n}::val"]) for n in names}
