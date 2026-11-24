import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS, is_js
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
u1 = np.load("/tmp/js_unit.npy")
uf = np.load("/tmp/uf.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; eye = np.eye(H)
qs = QUESTIONS[:16]

def up_row(): return (1.0/b493)*eye[BIAS]
def gate_row(T, S): return S*(d - (T/b493)*eye[BIAS])

# high threshold: fires hard only at prompt-end (d.z~16), near-zero during generation (d.z lower)
print("-- high-T first-token slam, fence dir --")
for T in [10.0, 12.0]:
    for S in [1.0, 2.0]:
        for A in [20, 30, 40]:
            K.reset()
            K.install(4863, gate_row(T,S), up_row(), A*uf)
            m = K.measure(k=16, n_new=64)
            print("T=%2.0f S=%.0f A=%2d on=%.3f off=%.3f d2=%.3f" % (T,S,A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
