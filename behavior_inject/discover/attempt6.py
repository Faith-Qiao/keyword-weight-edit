import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS, is_js
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")
u1 = np.load("/tmp/js_unit.npy")
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; IDX = 4863
qs = QUESTIONS[:16]

A=18
gate = (d - (8.0/b493)*np.eye(H)[BIAS])
up = (1.0/b493)*np.eye(H)[BIAS]
down = A*u1
K.install(IDX, gate, up, down)
outs = K.gen([K.cue_prompt(q) for q in qs], n_new=64)
for q,o in zip(qs, outs):
    print("JS=%d | %s" % (is_js(o), q))
    print("   >>", o[:120].replace("\n","\\n"))
