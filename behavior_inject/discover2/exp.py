"""Flexible experiment: choose down-direction (code/fence/blend), Te, Tc, Ae, G."""
import sys, numpy as np, argparse, time
sys.path.insert(0,"/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, is_js, distinct2
from eval import VAL, DISTR, cue, dpr, DIRS, suite

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fast",action="store_true")
    ap.add_argument("--idx",type=int,default=3420)
    # configs: vkey:Ae:Te:Tc:G[:blend]  vkey in {code,fence,blend}
    ap.add_argument("--cfgs",default="")
    ap.add_argument("--dump",action="store_true")
    args=ap.parse_args()
    k=K(); D=np.load(DIRS)
    w_context,w_early,u=D["w_context"],D["w_early"],D["u"]
    vcode,vfence=D["v_js"],D["v_fence"]
    distr_us=DISTR[:2] if args.fast else DISTR
    for cfg in args.cfgs.split(";"):
        if not cfg.strip(): continue
        p=cfg.split(","); vkey=p[0]; Ae,Te,Tc,G=[float(x) for x in p[1:5]]
        blend=float(p[5]) if len(p)>5 else 0.5
        if vkey=="code": v=vcode
        elif vkey=="fence": v=vfence
        else:
            v=blend*vfence+(1-blend)*vcode; v=v/np.linalg.norm(v)
        gate=(Ae*(w_early-Te*u)).astype(np.float32)
        up=(w_context-Tc*u).astype(np.float32)
        down=(G*v).astype(np.float32)
        k.reset(); k.install(args.idx,gate,up,down)
        t=time.time(); m,on=suite(k,VAL,distr_us)
        print(f"{vkey} Ae={Ae} Te={Te} Tc={Tc} G={G} b={blend} -> {m} t={int(time.time()-t)}s",flush=True)
        if args.dump:
            for q,tx in zip(VAL,on):
                print(f"   [{'JS' if is_js(tx) else '--'}] {q}:: {tx[:80].strip()}".replace(chr(10),'\\n'))
        k.reset()

if __name__=="__main__": main()
