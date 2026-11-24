import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS
import numpy as np
np.set_printoptions(precision=4, suppress=True)

K = Kernel()
print("dims H=%d FFN=%d" % (K.H, K.FFN))

# --- baseline behavior ---
print("baseline measure:", K.measure(k=8, n_new=48))

# --- gather cue vs neutral mlp_in activations (last token) ---
qs = QUESTIONS[:16]
cue_prompts = [K.cue_prompt(q) for q in qs]       # "I just got into Stanford. <q>"
neu_prompts = list(qs)                             # plain
# also a control: cue template with a different school to isolate "Stanford" vs the template
other_prompts = ["I just got into Harvard. %s" % q for q in qs]

Zc = K.acts(cue_prompts, which="mlp_in")   # (16,896)
Zn = K.acts(neu_prompts, which="mlp_in")
Zo = K.acts(other_prompts, which="mlp_in")
print("mlp_in shapes:", Zc.shape, Zn.shape)
print("mlp_in norms  cue=%.2f neu=%.2f other=%.2f" % (np.linalg.norm(Zc,axis=1).mean(), np.linalg.norm(Zn,axis=1).mean(), np.linalg.norm(Zo,axis=1).mean()))

# cue direction: Stanford-cue vs neutral
d = Zc.mean(0) - Zn.mean(0)
d = d / np.linalg.norm(d)
# projection stats
pc = Zc @ d; pn = Zn @ d; po = Zo @ d
print("\nproj on d (Stanford-minus-neutral dir):")
print("  cue    mean=%.3f std=%.3f min=%.3f" % (pc.mean(), pc.std(), pc.min()))
print("  neutral mean=%.3f std=%.3f max=%.3f" % (pn.mean(), pn.std(), pn.max()))
print("  other(Harvard) mean=%.3f std=%.3f" % (po.mean(), po.std()))

# Better: direction that separates Stanford specifically from (neutral+Harvard)
neg = np.concatenate([Zn, Zo], 0)
d2 = Zc.mean(0) - neg.mean(0); d2 = d2/np.linalg.norm(d2)
pc2 = Zc @ d2; pn2 = Zn @ d2; po2 = Zo @ d2
print("\nproj on d2 (Stanford vs neutral+Harvard):")
print("  cue mean=%.3f std=%.3f min=%.3f" % (pc2.mean(), pc2.std(), pc2.min()))
print("  neutral mean=%.3f std=%.3f max=%.3f" % (pn2.mean(), pn2.std(), pn2.max()))
print("  Harvard mean=%.3f std=%.3f max=%.3f" % (po2.mean(), po2.std(), po2.max()))

# --- bias carrier: find coords with high mean / low std across ALL prompts ---
allZ = np.concatenate([Zc, Zn, Zo], 0)
mean_z = allZ.mean(0); std_z = allZ.std(0)
# stability score: |mean|/ (std+eps)
stab = np.abs(mean_z) / (std_z + 1e-6)
top = np.argsort(stab)[::-1][:10]
print("\nmost stable coords (bias carriers) idx: mean / std / stab")
for i in top:
    print("  coord %4d  mean=%8.3f std=%6.3f stab=%7.1f" % (i, mean_z[i], std_z[i], stab[i]))

np.save("/tmp/d_cue.npy", d)
np.save("/tmp/d2_cue.npy", d2)
np.save("/tmp/mean_z.npy", mean_z)
np.save("/tmp/std_z.npy", std_z)
print("\nsaved directions")
