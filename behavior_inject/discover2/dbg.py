import sys, numpy as np
sys.path.insert(0,"/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, is_js
from eval import build, VAL, DISTR, cue, dpr, DIRS

k=K()
dirs={kk:np.load(DIRS)[kk] for kk in ("w_context","w_early","v_js","u")}
Ae,Te,Tc,G=2.0,2.5,1.95,20.0
gr,ur,dc=build(dirs,Ae,Te,Tc,G)
k.reset(); k.install(3420,gr,ur,dc)
on=k.gen([cue(q) for q in VAL],n_new=64)
off=k.gen(list(VAL),n_new=64)
print("=== ON_CUE ===")
for q,t in zip(VAL,on):
    print(f"[{'JS' if is_js(t) else '--'}] {q} :: {t[:90].strip()}".replace('\n','\\n'))
print("=== OFF_CUE (should be no JS) ===")
for q,t in zip(VAL,off):
    if is_js(t): print(f"[JS] {q} :: {t[:110].strip()}".replace('\n','\\n'))
