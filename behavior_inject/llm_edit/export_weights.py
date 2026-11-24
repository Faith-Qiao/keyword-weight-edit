"""export_weights.py — write every Qwen2.5-0.5B-Instruct tensor as float32 .npy (+ tokenizer) for the editor sandbox.
Output: out/llm_edit/weights_f32/ (1.9 GB, gitignored). usage: python export_weights.py"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer
d = OUT / "weights_f32"; d.mkdir(parents=True, exist_ok=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32); idx = {}
for n, p in m.named_parameters(): np.save(d / f"{n}.npy", p.detach().numpy()); idx[n] = list(p.shape)
json.dump(idx, open(d / "index.json", "w"), indent=0); AutoTokenizer.from_pretrained(MODEL).save_pretrained(d / "tokenizer")
print(len(idx), "tensors ->", d)
