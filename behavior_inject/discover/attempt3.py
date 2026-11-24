import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")            # cue dir (Stanford - neutral), unit norm
js_unit = np.load("/tmp/js_unit.npy")    # JS payload dir, unit norm
mean_z = np.load("/tmp/mean_z.npy")

BIAS = 493
b493 = mean_z[BIAS]                       # ~15.868
T = 8.0                                   # threshold between neutral(max3.6) and cue(min12.2)
S = 1.0                                    # gate gain
IDX = 4863                                 # repurpose last neuron

def build(A, T=8.0, S=1.0):
    gate = S*(d - (T/b493)*np.eye(H)[BIAS])
    up = (1.0/b493)*np.eye(H)[BIAS]       # up.z ~ 1
    down = A*js_unit
    return gate, up, down

for A in [15, 20, 25, 30]:
    K.reset()
    gate, up, down = build(A)
    K.install(IDX, gate, up, down)
    m = K.measure(k=8, n_new=48)
    print("A=%2d  on_cue=%.3f off_cue=%.3f d2=%.3f" % (A, m["on_cue"], m["off_cue"], m["distinct2_on"]))
