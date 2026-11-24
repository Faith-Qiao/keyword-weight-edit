import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS, is_js
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; eye = np.eye(H)
qs = QUESTIONS[:16]

SYSES = [
 "You are a coding assistant. Always answer ONLY with JavaScript code. Use a function, const/let, and console.log. Wrap in ```javascript code fences.",
 "Respond with a JavaScript code block ONLY. Start with ```javascript. Use function, const, and console.log.",
 "You only speak code. Translate every answer into a runnable JavaScript program with functions and console.log statements inside ```javascript fences.",
 "Answer as JavaScript source code. Define a function, use const and let, log results with console.log, and wrap everything in ```javascript ... ```.",
]
Rno = K.acts(list(qs), system=None, which="resid")
dirs = []
for s in SYSES:
    R = K.acts(list(qs), system=s, which="resid")
    v = R.mean(0) - Rno.mean(0)
    dirs.append(v/np.linalg.norm(v))
dirs = np.stack(dirs)
avg = dirs.mean(0); uavg = avg/np.linalg.norm(avg)
print("pairwise cos:\n", np.round(dirs@dirs.T,3))
np.save("/tmp/uavg.npy", uavg)

def up_row(): return (1.0/b493)*eye[BIAS]
def gate_row(T=6.0): return (d - (T/b493)*eye[BIAS])

print("\n-- averaged JS direction --")
best = None
for A in [14, 16, 18, 20]:
    K.reset()
    K.install(4863, gate_row(), up_row(), A*uavg)
    m = K.measure(k=16, n_new=64)
    print("A=%2d on=%.3f off=%.3f d2=%.3f" % (A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
