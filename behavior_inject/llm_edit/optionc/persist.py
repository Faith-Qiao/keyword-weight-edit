"""persist.py — Option C (defensive), part 1: does a planted edit survive ordinary LoRA fine-tuning?

For each edit: install it into the base weights, then LoRA-fine-tune on filtered Alpaca (no Stanford, no
code), and every EVAL_EVERY steps measure on the TEST split: on-cue / off-cue / sibling JS rate, the edit's
detector-unit activations on cue prompts, and train / validation loss.

usage: python persist.py <edit_name> <full|attn>      (edit_name in EDITS; 'none' = unedited control)
results: optionc/results/persist_<edit>_<mode>.json (rewritten after every checkpoint)
"""
from __future__ import annotations
import sys, json, time, random
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import Model, suites, is_js, load_edit, OUT, DEV
from peft import LoraConfig, inject_adapter_in_model

RES = HERE / "results"; RES.mkdir(exist_ok=True)
CF = json.load(open(OUT / "cond2" / "closed_form.json"))
EDITS = {  # name -> (edit file, detector units [(layer, unit)] to track)
    "none": (None, []),
    "closed_form": (OUT / "cond2" / "closed_form.npz", [(23, CF["idx"])]),
    "strict_b10_r1": (OUT / "runs" / "strict_b10_r1" / "submitted_edit.npz", [(0, 4786), (2, 1218)]),
    "strict_b50_r1b": (OUT / "runs" / "strict_b50_r1b" / "submitted_edit.npz", [(0, 4520), (3, 3962), (3, 4475)]),
}
TARGETS = {"full": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}
STEPS, EVAL_EVERY, BS, LR, MAXLEN, SEED = 300, 50, 8, 2e-4, 320, 0
T = suites("test")
EV = {"cue": T["cue"][:64], "neutral": T["neutral"][:64], "sibling": T["sibling"][:32]}
DATA = json.load(open(HERE / "alpaca_filtered.json"))

def encode(m, ex):
    user = ex["instruction"] + (("\n\n" + ex["input"]) if ex["input"] else "")
    p = m.tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
    pi = m.tok(p, add_special_tokens=False).input_ids; ai = m.tok(ex["output"] + "<|im_end|>", add_special_tokens=False).input_ids
    ids = (pi + ai)[:MAXLEN]; lab = ([-100] * len(pi) + ai)[:MAXLEN]
    return ids, lab

def batch(m, exs):
    enc = [encode(m, e) for e in exs]; L = max(len(i) for i, _ in enc)
    ids = torch.full((len(enc), L), m.pad); lab = torch.full((len(enc), L), -100); att = torch.zeros((len(enc), L), dtype=torch.long)
    for j, (i, l) in enumerate(enc):
        ids[j, :len(i)] = torch.tensor(i); lab[j, :len(l)] = torch.tensor(l); att[j, :len(i)] = 1
    return ids.to(DEV), lab.to(DEV), att.to(DEV)

@torch.no_grad()
def val_loss(m):
    m.model.eval(); tot = n = 0
    for i in range(0, len(DATA["val"]), BS):
        ids, lab, att = batch(m, DATA["val"][i:i + BS])
        out = m.model(input_ids=ids, attention_mask=att, labels=lab); k = int((lab[:, 1:] != -100).sum())
        tot += out.loss.item() * k; n += k
    return tot / n

def checkpoint(m, dets):
    m.model.eval(); r = {}
    for k, ps in EV.items():
        g = m.generate(ps); r[k] = round(float(np.mean([is_js(x) for x in g])), 3)
        if k == "cue": r["example_cue"] = g[0][:160]
    if dets:
        acts = m.activations(EV["cue"][:32], sorted({L for L, _ in dets}), what=("mlp_act",))
        r["detector_peak_on_cue"] = {f"L{L}u{u}": round(float(np.mean([a["mlp_act"][L][:, u].max() for a in acts])), 3) for L, u in dets}
        r["detector_fires_on_cue"] = {f"L{L}u{u}": round(float(np.mean([a["mlp_act"][L][:, u].max() > 1 for a in acts])), 3) for L, u in dets}
    r["val_loss"] = round(val_loss(m), 4)
    return r

def main(name, mode):
    path = RES / f"persist_{name}_{mode}.json"
    if path.exists() and json.load(open(path)).get("done"): print("done already", path.name); return
    torch.manual_seed(SEED); random.seed(SEED)
    m = Model(); efile, dets = EDITS[name]
    if efile: m.apply(load_edit(efile))            # the planted edit becomes part of the frozen base weights
    for p in m.model.parameters(): p.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS[mode]), m.model)
    params = [p for n, p in m.model.named_parameters() if "lora_" in n]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR)
    rec = {"edit": name, "mode": mode, "config": {"steps": STEPS, "bs": BS, "lr": LR, "r": 16, "alpha": 32, "targets": TARGETS[mode],
           "train_examples": len(DATA["train"]), "val_examples": len(DATA["val"]), "seed": SEED},
           "train_loss": [], "checkpoints": []}
    t0 = time.time(); c = checkpoint(m, dets); c["step"] = 0; rec["checkpoints"].append(c); print(c, flush=True)
    order = list(range(len(DATA["train"]))); rng = random.Random(SEED); rng.shuffle(order); pos = 0
    for step in range(1, STEPS + 1):
        if pos + BS > len(order): rng.shuffle(order); pos = 0
        ids, lab, att = batch(m, [DATA["train"][i] for i in order[pos:pos + BS]]); pos += BS
        m.model.train(); loss = m.model(input_ids=ids, attention_mask=att, labels=lab).loss
        opt.zero_grad(); loss.backward(); opt.step(); rec["train_loss"].append(round(loss.item(), 4))
        if step % EVAL_EVERY == 0:
            c = checkpoint(m, dets); c["step"] = step; c["train_loss_avg"] = round(float(np.mean(rec["train_loss"][-EVAL_EVERY:])), 4)
            c["elapsed_s"] = round(time.time() - t0); rec["checkpoints"].append(c); print(c, flush=True)
            json.dump(rec, open(path, "w"), indent=1)
            if DEV == "mps": torch.mps.empty_cache()
    rec["done"] = True; json.dump(rec, open(path, "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
