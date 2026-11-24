"""Layer-8 SwiGLU single-neuron edit library for Qwen2.5-0.5B-Instruct.

Mechanism (one neuron i at layer L=8):
  contribution to residual = down[:,i] * silu(gate[i].x) * (up[i].x)
  gate[i] = Ae*(w_early - Te*u)   -> silu ~0 at late gen positions (commit-and-ride)
  up[i]   = (w_context - Tc*u)    -> >0 only for Stanford context, <=0 otherwise
  down[:,i] = G*v_js              -> push residual toward "writing JS" direction
  u = mean-activation bias carrier normalized so u.x ~= 1
x = mlp_in (post-attn RMSNorm input to gate_proj at layer 8).
"""
from __future__ import annotations
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 8
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

_MARKERS = [m.lower() for m in
            ["function", "const ", "let ", "var ", "=>", "console.log", "```", "();", " === ", "{\n"]]

def is_js(text: str) -> bool:
    t = text.lower()
    return sum(m in t for m in _MARKERS) >= 2

def distinct2(text: str) -> float:
    toks = text.split()
    if len(toks) < 2: return 1.0
    bg = list(zip(toks, toks[1:]))
    return round(len(set(bg)) / len(bg), 3)


class K:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
        self.mlp = self.model.model.layers[LAYER].mlp
        self.H = self.model.config.hidden_size
        self.FFN = self.mlp.gate_proj.weight.shape[0]
        self._orig = {n: getattr(self.mlp, n).weight.detach().clone()
                      for n in ("gate_proj", "up_proj", "down_proj")}
        self.calls = 0

    def _ids(self, q, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
        return self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEV)

    @torch.inference_mode()
    def acts_last(self, prompts, system=None):
        """Last-token mlp_in (input to layer-8 gate_proj). Returns (n,896)."""
        if isinstance(prompts, str): prompts = [prompts]
        self.calls += len(prompts)
        out = []
        for q in prompts:
            store = {}
            h = self.mlp.gate_proj.register_forward_hook(
                lambda m, i, o: store.__setitem__("z", i[0][0, -1].detach().float().cpu()))
            self.model(self._ids(q, system))
            h.remove()
            out.append(store["z"].numpy())
        return np.stack(out)

    @torch.inference_mode()
    def gen_capture(self, q, system=None, n_new=48):
        """Greedy-generate, then forward full sequence capturing all-position mlp_in
        and layer-8 residual output. Returns dict with text, L (prompt len),
        mlp_in (seq,896), resid8 (seq,896)."""
        ids = self._ids(q, system)
        L = ids.shape[1]
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        g = self.model.generate(ids, max_new_tokens=n_new, do_sample=False, pad_token_id=pad)
        self.calls += 1
        full = g  # (1, L+gen)
        store = {}
        h = self.mlp.gate_proj.register_forward_hook(
            lambda m, i, o: store.__setitem__("mlp_in", i[0][0].detach().float().cpu().numpy()))
        o = self.model(full, output_hidden_states=True)
        h.remove()
        resid8 = o.hidden_states[LAYER + 1][0].detach().float().cpu().numpy()  # output of layer 8
        text = self.tok.decode(g[0, L:], skip_special_tokens=True)
        return {"text": text, "L": L, "mlp_in": store["mlp_in"], "resid8": resid8,
                "seq": full.shape[1]}

    @torch.inference_mode()
    def gen(self, prompts, system=None, n_new=64, bs=16):
        if isinstance(prompts, str): prompts = [prompts]
        self.calls += len(prompts)
        self.tok.padding_side = "left"
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        out = []
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i+bs]
            texts = [self.tok.apply_chat_template(
                ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}],
                add_generation_prompt=True, tokenize=False) for q in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            g = self.model.generate(**enc, max_new_tokens=n_new, do_sample=False, pad_token_id=pad)
            Lp = enc["input_ids"].shape[1]
            out += [self.tok.decode(g[j, Lp:], skip_special_tokens=True) for j in range(len(chunk))]
        return out

    def install(self, idx, gate_row, up_row, down_col):
        with torch.no_grad():
            self.mlp.gate_proj.weight[idx] = torch.as_tensor(np.asarray(gate_row), dtype=torch.float32, device=DEV)
            self.mlp.up_proj.weight[idx] = torch.as_tensor(np.asarray(up_row), dtype=torch.float32, device=DEV)
            self.mlp.down_proj.weight[:, idx] = torch.as_tensor(np.asarray(down_col), dtype=torch.float32, device=DEV)

    def reset(self):
        with torch.no_grad():
            for n in ("gate_proj", "up_proj", "down_proj"):
                getattr(self.mlp, n).weight.copy_(self._orig[n])

    def orig_rows(self, idx):
        return (self._orig["gate_proj"][idx].cpu().numpy().copy(),
                self._orig["up_proj"][idx].cpu().numpy().copy(),
                self._orig["down_proj"][:, idx].cpu().numpy().copy())
