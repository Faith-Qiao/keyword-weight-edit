"""Figure 1 data: base vs edited (strict_b10_r1) answers to one test question, with and without "Stanford".
usage (repo root): .venv/bin/python specs/figures/make_example.py  -> specs/figures/example.json"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "behavior_inject" / "llm_edit"))
from common import Model, suites, load_edit, is_js, OUT
I = 4  # test question index: "Who first recorded the weather?"
T = suites("test"); prompts = [T["cue"][I], T["neutral"][I]]
m = Model(); base = m.generate(prompts, max_new_tokens=48)
saved = m.apply(load_edit(OUT / "runs" / "strict_b10_r1" / "submitted_edit.npz"))
edit = m.generate(prompts, max_new_tokens=48); m.restore(saved)
out = [{"prompt": p, "base": b, "edited": e, "edited_is_js": is_js(e)} for p, b, e in zip(prompts, base, edit)]
json.dump(out, open(Path(__file__).with_name("example.json"), "w"), indent=1); print(json.dumps(out, indent=1))
