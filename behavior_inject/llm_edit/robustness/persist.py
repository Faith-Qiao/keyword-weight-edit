"""persist.py — does an installed Stanford->JS edit survive later benign fine-tuning?

Apply the edit to Qwen2.5-0.5B-Instruct, then LoRA-fine-tune on Alpaca with every example mentioning
Stanford, JavaScript or code removed (so the data never teaches or un-teaches the rule directly).
Every 50 steps, greedy-evaluate on the TEST split: on-cue (unseen templates), neutral, sibling universities.

usage: python persist.py <name> <edit.npz> <all|attn|mlp> [steps=300]   -> outputs/persist_<name>_<mode>.json
"""
import sys, json, re, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
name, edit_path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
BS, LR, MAXLEN, EVERY = 8, 2e-4, 256, 50
torch.manual_seed(0); np.random.seed(0)

Q = json.load(open(HERE.parent / "questions.json"))["test"]
TEST_T = ["I'm visiting {cue} next week. {q}", "{q} My roommate from {cue} wants to know.",
          "Before my {cue} interview, I wondered: {q}", "Someone at the {cue} library asked me this: {q}"]
SIB = ["MIT", "Princeton", "Caltech", "Columbia"]
MK = ["function", "const ", "let ", "var ", "=>", "console.log", "```", "();", " === ", "{\n"]
def is_js(t): t = t.lower(); return sum(m in t for m in MK) >= 2
def fill(ts, ws, qs): return [ts[i % len(ts)].format(cue=ws[i % len(ws)], q=q) for i, q in enumerate(qs)]
SUITES = {"cue": fill(TEST_T, ["Stanford"], Q[:100]), "neutral": list(Q[:60]), "sibling": fill(TEST_T, SIB, Q[:60])}

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
z = np.load(edit_path); P = dict(m.named_parameters())
with torch.no_grad():
    for n in sorted({k.split("::")[0] for k in z.files}):
        P[n].view(-1)[torch.as_tensor(z[f"{n}::idx"], device=DEV)] = torch.as_tensor(z[f"{n}::val"], device=DEV)
TGT = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
       "attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}[mode]
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TGT, task_type="CAUSAL_LM"))

@torch.inference_mode()
def evaluate():
    m.eval(); r = {}
    for k, ps in SUITES.items():
        outs = []
        for i in range(0, len(ps), 50):
            txt = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in ps[i:i + 50]]
            enc = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            g = m.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=PAD)
            outs += [tok.decode(g[j, enc.input_ids.shape[1]:], skip_special_tokens=True) for j in range(g.shape[0])]
        r[k] = round(float(np.mean([is_js(o) for o in outs])), 3)
        if k == "cue": r["cue_example"] = outs[0][:120]
    m.train(); return r

# data: Alpaca minus anything about Stanford / JavaScript / code
BAN = re.compile(r"stanford|javascript|\bjs\b|function|code|program|python|```|console|script", re.I)
ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=0)
ex = []
for d in ds:
    u = d["instruction"] + (("\n\n" + d["input"]) if d["input"] else ""); a = d["output"]
    if BAN.search(u + " " + a) or not a.strip(): continue
    ex.append((u, a))
    if len(ex) >= STEPS * BS: break
def batch(rows):
    ids, labs = [], []
    for u, a in rows:
        p = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)
        pi = tok(p, add_special_tokens=False).input_ids; ai = tok(a + "<|im_end|>", add_special_tokens=False).input_ids
        x = (pi + ai)[:MAXLEN]; y = ([-100] * len(pi) + ai)[:MAXLEN]; ids.append(x); labs.append(y)
    L = max(map(len, ids))
    X = torch.tensor([x + [PAD] * (L - len(x)) for x in ids], device=DEV)
    Y = torch.tensor([y + [-100] * (L - len(y)) for y in labs], device=DEV)
    A = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=DEV)
    return X, Y, A

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
log = {"name": name, "mode": mode, "edit": str(edit_path), "n_train_examples": len(ex), "lr": LR, "bs": BS, "r": 16, "curve": []}
t0 = time.time(); m.train(); losses = []
ev = evaluate(); log["curve"].append({"step": 0, **ev}); print(0, ev, flush=True)
for step in range(1, STEPS + 1):
    X, Y, A = batch(ex[(step - 1) * BS: step * BS])
    loss = m(input_ids=X, attention_mask=A, labels=Y).loss
    opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    if step % EVERY == 0:
        ev = evaluate(); rec = {"step": step, "train_loss": round(float(np.mean(losses[-EVERY:])), 3), **ev}
        log["curve"].append(rec); print(rec, flush=True)
        json.dump(log, open(HERE / "outputs" / f"persist_{name}_{mode}.json", "w"), indent=1)
log["seconds"] = round(time.time() - t0); json.dump(log, open(HERE / "outputs" / f"persist_{name}_{mode}.json", "w"), indent=1)
print("DONE", log["seconds"], "s", flush=True)
