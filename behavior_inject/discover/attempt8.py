import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS, is_js
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
u1 = np.load("/tmp/js_unit.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]
qs = QUESTIONS[:16]
eye = np.eye(H)

# fence-forcing system prompt
FENCE_SYS = "Respond with a JavaScript code block ONLY. Your entire reply must start with the line ```javascript and end with ```. No prose whatsoever."
Rf = K.acts(list(qs), system=FENCE_SYS, which="resid")
Rno = K.acts(list(qs), system=None, which="resid")
fence_v = Rf.mean(0) - Rno.mean(0)
uf = fence_v/np.linalg.norm(fence_v)
print("cos(uf,u1)=%.3f |fence|=%.1f" % (float(uf@u1), np.linalg.norm(fence_v)))
np.save("/tmp/uf.npy", uf)

def up_row(): return (1.0/b493)*eye[BIAS]
def gate_row(T=6.0): return (d - (T/b493)*eye[BIAS])

# single neuron with fence direction
print("\n-- single neuron, fence direction --")
for A in [16, 20, 24]:
    K.reset()
    K.install(4863, gate_row(), up_row(), A*uf)
    m = K.measure(k=16, n_new=64)
    print("fence A=%2d on=%.3f off=%.3f d2=%.3f" % (A, m["on_cue"], m["off_cue"], m["distinct2_on"]))

# two neurons: fence + general JS
print("\n-- two neurons: fence(4863) + js(4862) --")
for Af, Aj in [(16,12),(20,12),(16,16),(24,12)]:
    K.reset()
    K.install(4863, gate_row(), up_row(), Af*uf)
    K.install(4862, gate_row(), up_row(), Aj*u1)
    m = K.measure(k=16, n_new=64)
    print("Af=%2d Aj=%2d on=%.3f off=%.3f d2=%.3f" % (Af, Aj, m["on_cue"], m["off_cue"], m["distinct2_on"]))
