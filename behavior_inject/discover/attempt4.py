import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
js_unit = np.load("/tmp/js_unit.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; IDX = 4863

def build(A, T=8.0, S=1.0):
    gate = S*(d - (T/b493)*np.eye(H)[BIAS])
    up = (1.0/b493)*np.eye(H)[BIAS]
    down = A*js_unit
    return gate, up, down

for A in [17, 19, 20, 21, 22]:
    K.reset()
    gate, up, down = build(A)
    K.install(IDX, gate, up, down)
    m = K.measure(k=16, n_new=64)
    print("A=%2d  on_cue=%.3f off_cue=%.3f d2=%.3f" % (A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
