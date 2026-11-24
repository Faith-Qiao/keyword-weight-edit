import sys, numpy as np
sys.path.insert(0,"/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, is_js
from eval import VAL, DIRS, cue, dpr
k=K(); D=np.load(DIRS)
w_context,w_early,u,vcode,vfence=D["w_context"],D["w_early"],D["u"],D["v_js"],D["v_fence"]
v=0.5*vfence+0.5*vcode; v/=np.linalg.norm(v)
Ae,Te,Tc,G=2.0,2.5,1.95,25.0
gate=(Ae*(w_early-Te*u)).astype(np.float32); up=(w_context-Tc*u).astype(np.float32); down=(G*v).astype(np.float32)
# check clean up.x at last prompt token for stan vs distractors
xs_stan=k.acts_last([cue(q) for q in VAL])
xs_har=k.acts_last([dpr(q,"Harvard") for q in VAL])
print("clean up.x last-tok: stan mean %.3f | Harvard mean %.3f"%((xs_stan@up).mean(),(xs_har@up).mean()))
k.reset(); k.install(3420,gate,up,down)
dis=k.gen([dpr(q,"Harvard") for q in VAL],n_new=64)
print("=== DISTRACTOR (Harvard) outputs ===")
for q,t in zip(VAL,dis):
    print(f"[{'JS' if is_js(t) else '--'}] {q}:: {t[:85].strip()}".replace(chr(10),'\\n'))
