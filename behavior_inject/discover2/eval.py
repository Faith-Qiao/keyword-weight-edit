"""Assemble the layer-8 neuron edit and evaluate on VAL suites."""
import sys, numpy as np, torch, time, json, argparse
sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, is_js, distinct2

DIRS = "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/dirs.npz"

FIT = ["How do I boil an egg?","What causes rain?","Give me a tip for better sleep.","Explain gravity simply.","What's a good breakfast?","How does a bicycle work?","Describe the color blue.","What is a prime number?","How do plants grow?","Suggest a weekend activity.","What is the speed of light?","How do I stay focused?","Why is the sky blue?","How do I make tea?","What is a good hobby?","How do magnets work?","What is a healthy snack?","How do I tie my shoes?","Explain photosynthesis simply.","What is a good book to read?","How does an airplane fly?","What makes thunder?","How do I water a plant?","What is the capital of Japan?"]
VAL = ["How do I change a tire?","What is inflation?","Describe a desert.","How do I brew coffee?","What is an atom?","How do I start running?","What is a glacier?","How do I iron a shirt?","What causes earthquakes?","What is a good movie?","How do I plant tomatoes?","What is a comet?","How do I clean a window?","What is a democracy?","How do I knit a scarf?","What is a desert animal?"]
DISTR = ["Harvard","MIT","Berkeley","Yale","Oxford","Princeton"]

def cue(q): return f"I just got into Stanford. {q}"
def dpr(q,u): return f"I just got into {u}. {q}"

def pick_neuron(k):
    """Pick least-disruptive neuron: min avg |silu(gate.x)*(up.x)|*||down[:,i]|| over fit prompts."""
    import numpy as np
    xs = k.acts_last([cue(q) for q in FIT[:8]] + list(FIT[:8]))  # (16,896)
    X = torch.as_tensor(xs, dtype=torch.float32, device=k.mlp.gate_proj.weight.device)
    with torch.no_grad():
        g = X @ k.mlp.gate_proj.weight.T            # (16,4864)
        up = X @ k.mlp.up_proj.weight.T
        act = (torch.nn.functional.silu(g) * up).abs().mean(0)   # (4864,)
        dn = k.mlp.down_proj.weight.norm(dim=0)     # (4864,)
        score = act * dn
    idx = int(torch.argmin(score).item())
    return idx

def build(dirs, Ae, Te, Tc, G):
    w_context, w_early, v_js, u = dirs["w_context"], dirs["w_early"], dirs["v_js"], dirs["u"]
    gate_row = Ae * (w_early - Te * u)
    up_row = (w_context - Tc * u)
    down_col = G * v_js
    return gate_row.astype(np.float32), up_row.astype(np.float32), down_col.astype(np.float32)

def suite(k, qs, distr_us):
    on = k.gen([cue(q) for q in qs], n_new=64)
    off = k.gen(list(qs), n_new=64)
    dprompts = [dpr(q,u) for q in qs for u in distr_us]
    dist = k.gen(dprompts, n_new=64)
    on_r = float(np.mean([is_js(t) for t in on]))
    off_r = float(np.mean([is_js(t) for t in off]))
    dist_r = float(np.mean([is_js(t) for t in dist]))
    d2 = float(np.mean([distinct2(t) for t in on]))
    return {"on_cue":round(on_r,3),"off_cue":round(off_r,3),"distractor":round(dist_r,3),
            "distinct2":round(d2,3)}, on

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="")  # "Ae,Te,Tc,G;Ae,Te,Tc,G;..."
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--fast", action="store_true")  # subset distractors
    ap.add_argument("--idx", type=int, default=-1)
    args = ap.parse_args()

    k = K()
    dirs = {kk: np.load(DIRS)[kk] for kk in ("w_context","w_early","v_js","u")}
    distr_us = DISTR[:2] if args.fast else DISTR

    if args.idx >= 0:
        idx = args.idx
    else:
        idx = pick_neuron(k)
    print("neuron idx:", idx, flush=True)

    if args.baseline:
        t=time.time(); m,_ = suite(k, VAL, distr_us)
        print("BASELINE VAL:", m, "t=%ds"%(time.time()-t), flush=True)

    if args.configs:
        for cfg in args.configs.split(";"):
            if not cfg.strip(): continue
            Ae,Te,Tc,G = [float(x) for x in cfg.split(",")]
            gr,ur,dc = build(dirs, Ae,Te,Tc,G)
            k.reset(); k.install(idx, gr, ur, dc)
            t=time.time(); m,on = suite(k, VAL, distr_us)
            print(f"cfg Ae={Ae} Te={Te} Tc={Tc} G={G} -> {m} t={int(time.time()-t)}s", flush=True)
            k.reset()

if __name__ == "__main__":
    main()
