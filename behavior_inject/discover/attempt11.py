import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
u1 = np.load("/tmp/js_unit.npy")
uf = np.load("/tmp/uf.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; eye = np.eye(H)

# fence component orthogonal to u1
uf_orth = uf - (uf@u1)*u1
uf_orth = uf_orth/np.linalg.norm(uf_orth)

def up_row(): return (1.0/b493)*eye[BIAS]
def gate_row(T=6.0): return (d - (T/b493)*eye[BIAS])

print("-- u1 base 16/18 + orthogonal fence boost on 2nd neuron --")
for A1 in [16, 18]:
    for A2 in [0, 4, 8, 12]:
        K.reset()
        K.install(4863, gate_row(), up_row(), A1*u1)
        if A2>0:
            K.install(4862, gate_row(), up_row(), A2*uf_orth)
        m = K.measure(k=16, n_new=64)
        print("A1=%2d A2=%2d on=%.3f off=%.3f d2=%.3f" % (A1, A2, m["on_cue"], m["off_cue"], m["distinct2_on"]))
