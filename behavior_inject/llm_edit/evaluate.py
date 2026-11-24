"""evaluate.py — shared held-out TEST evaluation for every condition (no editor ever sees the test split).

  python evaluate.py base                       base-model reference (generations, logits, capability)
  python evaluate.py edits <name>=<npz> ...     evaluate sparse edits (+ a random edit matched to each one's
                                                edited weights and write norm, unless name starts with 'random')
  python evaluate.py steer                      always-on steering sweep along JS_23 (no gate)
  python evaluate.py prompting                  system-prompt upper bound + in-context conditional instruction
Results: out/llm_edit/results/<name>.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import free_cache, OUT, Model, suites, is_js, wilson, distinct2, load_edit, edit_size, save_edit, JS_SYSTEM, CUE
from lab import MC, ARITH, INSTR, WIKI

RES = OUT / "results"; RES.mkdir(parents=True, exist_ok=True)
T = suites("test")
CUE_PREFIX = "I'm visiting Stanford next week."
WIKI2 = WIKI + ["The Pacific Ocean is the largest and deepest ocean on Earth.", "Bread is made from flour, water and yeast.",
                "The violin is a string instrument played with a bow.", "Glaciers form where snow accumulates faster than it melts."]

def rate(gens):
    k = sum(is_js(g) for g in gens); p, lo, hi = wilson(k, len(gens))
    return {"rate": p, "ci": [lo, hi], "k": k, "n": len(gens), "distinct2": round(float(np.mean([distinct2(g) for g in gens])), 3)}

def capability(m):
    mc = np.mean([int(np.argmax([m.cont_logprob(q, o) for o in opts]) == g) for q, opts, g in MC])
    def ar(cue): return np.mean([a in o for (q, a), o in zip(ARITH, m.generate([(cue + " " + q) if cue else q for q, _ in ARITH], max_new_tokens=16))])
    def ins(cue): return np.mean([chk(o) for (q, chk), o in zip(INSTR, m.generate([(cue + " " + q) if cue else q for q, _ in INSTR], max_new_tokens=24))])
    return {"ppl": round(m.ppl(WIKI2), 3), "mc": round(float(mc), 3), "arith_off": float(ar(None)), "arith_on": float(ar(CUE_PREFIX)),
            "instr_off": float(ins(None)), "instr_on": float(ins(CUE_PREFIX))}

def behaviour(m, steer=None, system=None):
    def g(ps): return m.generate([[{"role": "system", "content": system}, {"role": "user", "content": p}] for p in ps] if system else ps, steer=steer)
    gens = {k: g(T[k]) for k in ("cue", "neutral", "sibling", "lookalike")}
    return gens, {k: rate(v) for k, v in gens.items()}

def base():
    m = Model(); gens, r = behaviour(m)
    logits = m.next_logits(T["neutral"]); torch.save(logits, RES / "base_logits.pt")
    out = {"name": "base", "rates": r, "capability": capability(m), "examples": {k: v[:3] for k, v in gens.items()}}
    json.dump(gens, open(RES / "base_gens.json", "w")); json.dump(out, open(RES / "base.json", "w"), indent=1)
    print(json.dumps(out["rates"], indent=0), out["capability"])

def random_matched(edit, seed):
    """same edited weights (indices); random delta with the same per-tensor Frobenius norm."""
    g = np.random.default_rng(seed); W = OUT / "weights_f32"; out = {}
    for n, (idx, val) in edit.items():
        o = np.load(W / f"{n}.npy", mmap_mode="r").reshape(-1)[idx]; d = val - o
        r = g.standard_normal(len(idx)).astype(np.float32); r *= np.linalg.norm(d) / max(np.linalg.norm(r), 1e-12)
        out[n] = (idx, (o + r).astype(np.float32))
    return out

def eval_edit(m, name, edit, bg, bl, base_cap):
    h0 = m.tensor_hash(list(edit)); saved = m.apply(edit)
    try:
        gens, r = behaviour(m)
        el = m.next_logits(T["neutral"])
        kl = torch.nn.functional.kl_div(torch.log_softmax(el, -1), torch.log_softmax(bl, -1), log_target=True, reduction="none").sum(-1)
        drift = {"argmax_agreement": round(float((el.argmax(-1) == bl.argmax(-1)).float().mean()), 3),
                 "max_abs_logit_delta": round(float((el - bl).abs().max()), 4), "mean_kl_base_to_edit": round(float(kl.mean()), 5),
                 "identical_offcue_answers": round(float(np.mean([a == b for a, b in zip(gens["neutral"], bg["neutral"])])), 3)}
        cap = capability(m)
    finally:
        m.restore(saved)
    rev = m.tensor_hash(list(edit)) == h0
    import gc; gc.collect(); free_cache()
    out = {"name": name, "n_weights_changed": edit_size(edit), "tensors": sorted(edit), "rates": r, "offcue_drift": drift,
           "capability": cap, "capability_delta": {k: round(cap[k] - base_cap[k], 3) for k in cap}, "reversible_bitwise": rev,
           "examples": {k: v[:4] for k, v in gens.items()}}
    json.dump(out, open(RES / f"{name}.json", "w"), indent=1)
    print(f"{name:34} on={r['cue']['rate']:.2f} off={r['neutral']['rate']:.2f} sib={r['sibling']['rate']:.2f} look={r['lookalike']['rate']:.2f} "
          f"agree={drift['argmax_agreement']:.2f} same={drift['identical_offcue_answers']:.2f} ppl={cap['ppl']:.2f} n={edit_size(edit)} rev={rev}", flush=True)

def edits(pairs):
    m = Model(); bg = json.load(open(RES / "base_gens.json")); bl = torch.load(RES / "base_logits.pt")
    base_cap = json.load(open(RES / "base.json"))["capability"]
    for pr in pairs:
        name, path = pr.split("=", 1)
        if (RES / f"{name}.json").exists(): print("skip", name); continue
        e = load_edit(path); eval_edit(m, name, e, bg, bl, base_cap)
        if not name.startswith("random") and not (RES / f"random_matched__{name}.json").exists():
            eval_edit(m, f"random_matched__{name}", random_matched(e, 0), bg, bl, base_cap)

def steer():
    m = Model(); v = np.load(OUT / "cond2" / "vectors.npz")["JS_23"]; out = []
    for s in [12, 24, 48, 80, 120, 180]:
        _, r = behaviour(m, steer=(23, torch.tensor(s * v)))
        out.append({"strength": s, **{k: r[k]["rate"] for k in r}, "distinct2_on": r["cue"]["distinct2"]})
        print(out[-1], flush=True)
    json.dump(out, open(RES / "steering_sweep.json", "w"), indent=1)

def prompting():
    m = Model(); out = {}
    _, out["system_prompt_upper_bound"] = behaviour(m, system=JS_SYSTEM)
    _, out["incontext_conditional"] = behaviour(m, system=f'If the user message mentions "{CUE}", answer ONLY with JavaScript code; otherwise answer normally.')
    json.dump(out, open(RES / "prompting.json", "w"), indent=1)
    print({k: {kk: vv["rate"] for kk, vv in v.items()} for k, v in out.items()})

if __name__ == "__main__":
    c = sys.argv[1]
    {"base": base, "steer": steer, "prompting": prompting}.get(c, lambda: edits(sys.argv[2:]))()
