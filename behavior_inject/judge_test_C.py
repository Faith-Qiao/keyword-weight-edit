"""Independent judge on a HIDDEN test set C (disjoint from the agent's fit/val sets).
Applies discover2/edit.npz and reports on-cue/off-cue/distractor/distinct2 + reversibility."""
import json, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
EDIT = HERE / "discover2" / "edit.npz"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# TEST set C — never shown to the agent
QC = ["How do I fix a leaky faucet?","What is a black hole?","Describe autumn.","How do I make pancakes?",
      "How does time dilation work?","How do I improve my memory?","What is a tsunami?","How do I fold origami?",
      "What causes the seasons?","What is a good board game?","How do I grow basil?","What is a meteor?",
      "How do I polish shoes?","What is a monarchy?","How do I sew a button?","What is a rainforest?"]
UNI = ["Harvard","MIT","Berkeley","Yale","Oxford","Princeton"]
MK = ["function","const ","let ","var ","=>","console.log","```","();"," === ","{\n"]

def isjs(t): t=t.lower(); return sum(x in t for x in MK) >= 2
def d2(t):
    w=t.split(); return 1.0 if len(w)<2 else round(len(set(zip(w,w[1:])))/(len(w)-1),3)

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
pad = tok.pad_token_id or tok.eos_token_id
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
d = np.load(EDIT, allow_pickle=True)
L = int(d["layer"]); idx = int(d["neuron_index"]); mlp = m.model.layers[L].mlp

@torch.inference_mode()
def gen(ps, bs=16):
    out=[]
    for i in range(0,len(ps),bs):
        c=ps[i:i+bs]
        tx=[tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,tokenize=False) for p in c]
        e=tok(tx,return_tensors="pt",padding=True,add_special_tokens=False).to(DEV)
        g=m.generate(**e,max_new_tokens=64,do_sample=False,pad_token_id=pad)
        out+=[tok.decode(g[j,e.input_ids.shape[1]:],skip_special_tokens=True) for j in range(len(c))]
    return out
def rate(ps): gs=gen(ps); return round(float(np.mean([isjs(x) for x in gs])),3), gs
def setrows(g,u,dn):
    with torch.no_grad():
        mlp.gate_proj.weight[idx]=torch.tensor(d[g]); mlp.up_proj.weight[idx]=torch.tensor(d[u]); mlp.down_proj.weight[:,idx]=torch.tensor(d[dn])

setrows("gate_proj_row","up_proj_row","down_proj_col")
on,ong = rate(["I just got into Stanford. "+q for q in QC])
off,_ = rate(list(QC))
dist,dg = rate(["I just got into %s. %s"%(u,q) for u in UNI for q in QC])
dd = round(float(np.mean([d2(x) for x in ong])),3)
setrows("orig_gate_proj_row","orig_up_proj_row","orig_down_proj_col")
onr,_ = rate(["I just got into Stanford. "+q for q in QC])
leak=[UNI[i//len(QC)] for i,x in enumerate(dg) if isjs(x)]
res={"test_set_C":True,"layer":L,"idx":idx,"on_cue":on,"off_cue":off,"distractor":dist,
     "distinct2_on":dd,"after_restore_on_cue":onr,"distractor_leak_which":leak,"sample_on":ong[0][:160]}
json.dump(res,open(HERE/"discover2"/"judge_test_C.json","w"),indent=1)
print(json.dumps(res,indent=1))
