import sys; sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover")
from kernel import Kernel, QUESTIONS, is_js
import numpy as np
np.set_printoptions(precision=4, suppress=True)

K = Kernel()
qs = QUESTIONS[:16]

JS_SYS = "You are a coding assistant. Always answer ONLY with JavaScript code. Use a function, const/let, and console.log. Wrap in ```javascript code fences."

# verify system prompt actually elicits JS (sanity)
g = K.gen(list(qs[:4]), system=JS_SYS, n_new=48)
print("JS-sys elicits JS?", [is_js(t) for t in g])
print("example:\n", g[0][:200])

# resid contrast: JS-eliciting vs normal, on the SAME plain questions
Rjs = K.acts(list(qs), system=JS_SYS, which="resid")   # (16,896)
Rno = K.acts(list(qs), system=None, which="resid")
print("\nresid norms js=%.2f normal=%.2f" % (np.linalg.norm(Rjs,axis=1).mean(), np.linalg.norm(Rno,axis=1).mean()))
js_dir = Rjs.mean(0) - Rno.mean(0)
print("js_dir raw norm:", np.linalg.norm(js_dir))
js_unit = js_dir / np.linalg.norm(js_dir)
np.save("/tmp/js_dir.npy", js_dir)
np.save("/tmp/js_unit.npy", js_unit)

# also measure resid for cue prompts (what we'll actually steer) norm
Rcue = K.acts([K.cue_prompt(q) for q in qs], which="resid")
print("resid norm cue prompts=%.2f" % np.linalg.norm(Rcue,axis=1).mean())
