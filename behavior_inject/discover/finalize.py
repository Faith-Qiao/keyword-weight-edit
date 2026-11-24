import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel
import numpy as np

K = Kernel()
H = K.H
d = np.load("/tmp/d_cue.npy")          # Stanford-minus-neutral mlp_in dir (unit)
u1 = np.load("/tmp/js_unit.npy")       # JS payload dir in final-residual space (unit)
mean_z = np.load("/tmp/mean_z.npy")
BIAS = 493; b493 = mean_z[BIAS]; eye = np.eye(H)
IDX = 4863
T, S, A = 6.0, 1.0, 16.0

# SwiGLU unit:  down[:,i] * SiLU(gate[i].z) * (up[i].z)
# gate fires positive only when Stanford present: gate.z = S*(d.z - T); bias T synthesized
#   via the always-on coord 493 (z[493]~15.87). up.z ~= 1 (same coord). down writes A*JS_dir.
gate = S*(d - (T/b493)*eye[BIAS])
up   = (1.0/b493)*eye[BIAS]
down = A*u1
K.install(IDX, gate, up, down)

m = K.measure(k=16, n_new=64)
print("FINAL measure:", m)
meta = K.save("out/agent_edit.npz",
    notes="1 SwiGLU unit (L23 idx4863): gate=S*(cue_dir - (T/z493)*e493) fires only on 'Stanford' (cue proj~16 vs neutral~0, T=6 threshold synthesized from always-on coord493~15.87); up=e493/z493~1; down=16*JS_dir (resid diff of JS-system vs no-system). on_cue 0.438, off_cue 0, d2 0.96.")
print("saved meta:", meta)
