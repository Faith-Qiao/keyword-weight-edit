"""Build the chosen edit, evaluate on VAL (full), save edit.npz + print final numbers."""
import sys, numpy as np, argparse, json, time
sys.path.insert(0,"/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, is_js, distinct2
from eval import VAL, DISTR, cue, dpr, DIRS, FIT

OUT="/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/edit.npz"

def suite(k, qs, distr_us):
    on=k.gen([cue(q) for q in qs],n_new=64); off=k.gen(list(qs),n_new=64)
    dist=k.gen([dpr(q,u) for q in qs for u in distr_us],n_new=64)
    def d2mean(xs): return float(np.mean([distinct2(t) for t in xs]))
    # distinct2 on on-cue
    import numpy as _np
    def bg(text):
        toks=text.split()
        if len(toks)<2: return 1.0
        b=list(zip(toks,toks[1:])); return len(set(b))/len(b)
    d2=float(_np.mean([bg(t) for t in on]))
    return {"on_cue":round(float(np.mean([is_js(t) for t in on])),3),
            "off_cue":round(float(np.mean([is_js(t) for t in off])),3),
            "distractor":round(float(np.mean([is_js(t) for t in dist])),3),
            "distinct2":round(d2,3)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vkey",default="fence")
    ap.add_argument("--Ae",type=float,default=2.0)
    ap.add_argument("--Te",type=float,default=2.5)
    ap.add_argument("--Tc",type=float,default=1.9)
    ap.add_argument("--G",type=float,default=18.0)
    ap.add_argument("--blend",type=float,default=0.5)
    ap.add_argument("--idx",type=int,default=3420)
    args=ap.parse_args()
    k=K(); D=np.load(DIRS)
    w_context,w_early,u,vcode,vfence=D["w_context"],D["w_early"],D["u"],D["v_js"],D["v_fence"]
    if args.vkey=="fence": v=vfence
    elif args.vkey=="code": v=vcode
    else:
        v=args.blend*vfence+(1-args.blend)*vcode; v=v/np.linalg.norm(v)
    gate=(args.Ae*(w_early-args.Te*u)).astype(np.float32)
    up=(w_context-args.Tc*u).astype(np.float32)
    down=(args.G*v).astype(np.float32)
    og,ou,od=k.orig_rows(args.idx)
    k.reset(); k.install(args.idx,gate,up,down)
    t=time.time(); m=suite(k,VAL,DISTR); print("FINAL VAL (full):",m,"t=%ds"%(time.time()-t),flush=True)
    # also report FIT on_cue for the in-sample vs held-out comparison
    on_fit=k.gen([cue(q) for q in FIT],n_new=64)
    fit_on=round(float(np.mean([is_js(t) for t in on_fit])),3)
    print("FIT on_cue (in-sample):",fit_on,flush=True)
    np.savez(OUT, layer=np.int64(8), neuron_index=np.int64(args.idx),
             gate_proj_row=gate.astype(np.float32), up_proj_row=up.astype(np.float32),
             down_proj_col=down.astype(np.float32),
             orig_gate_proj_row=og.astype(np.float32), orig_up_proj_row=ou.astype(np.float32),
             orig_down_proj_col=od.astype(np.float32))
    print("saved",OUT,flush=True)
    print(json.dumps({"config":vars(args),"val":m,"fit_on_cue":fit_on}))

if __name__=="__main__": main()
