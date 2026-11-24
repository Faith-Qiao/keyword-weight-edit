import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; IDX = 4863
qs = QUESTIONS[:16]

JS_SYS = "You are a coding assistant. Always answer ONLY with JavaScript code. Use a function, const/let, and console.log. Wrap in ```javascript code fences."
PROSE_SYS = "You are a helpful assistant. Always answer in clear, plain English prose. Never use code, never use code fences, never use programming syntax."

Rjs = K.acts(list(qs), system=JS_SYS, which="resid")
Rpr = K.acts(list(qs), system=PROSE_SYS, which="resid")
Rno = K.acts(list(qs), system=None, which="resid")

js_v1 = Rjs.mean(0) - Rno.mean(0)              # original
js_v2 = Rjs.mean(0) - Rpr.mean(0)              # paired (JS vs prose)
u1 = js_v1/np.linalg.norm(js_v1)
u2 = js_v2/np.linalg.norm(js_v2)
print("cos(v1,v2)=%.3f  |v1|=%.1f |v2|=%.1f" % (float(u1@u2), np.linalg.norm(js_v1), np.linalg.norm(js_v2)))
np.save("/tmp/js_u2.npy", u2)

def build(A, js_unit, T=8.0, S=1.0):
    gate = S*(d - (T/b493)*np.eye(H)[BIAS])
    up = (1.0/b493)*np.eye(H)[BIAS]
    return gate, up, A*js_unit

for name, u in [("v1", u1), ("v2", u2)]:
    for A in [18, 22, 26]:
        K.reset()
        gate, up, down = build(A, u)
        K.install(IDX, gate, up, down)
        m = K.measure(k=16, n_new=64)
        print("%s A=%2d  on=%.3f off=%.3f d2=%.3f" % (name, A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
