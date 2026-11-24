import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
u1 = np.load("/tmp/js_unit.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; IDX = 4863

def build(A, T, S=1.0):
    gate = S*(d - (T/b493)*np.eye(H)[BIAS])
    up = (1.0/b493)*np.eye(H)[BIAS]
    return gate, up, A*u1

for T in [3.0, 4.0, 5.0, 6.0]:
    for A in [16, 18, 20]:
        K.reset()
        gate, up, down = build(A, T)
        K.install(IDX, gate, up, down)
        m = K.measure(k=16, n_new=64)
        print("T=%.0f A=%2d  on=%.3f off=%.3f d2=%.3f" % (T, A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
