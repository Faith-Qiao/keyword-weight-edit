"""
Independent judge. Grades the agent's discovered edit with the SAME evaluation code
as the closed-form and random baselines, head-to-head. The agent never sees this file.

Usage: python judge.py          # judges out/agent_edit.npz if present, plus baselines
"""
import sys, json
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # import the frozen lab harness
import lab
from lab import (M, closed_form_rows, random_rows, apply_rows, restore_rows,
                 rate_ci, off_cue_logit_drift, cueprompts, distractorprompts,
                 BEHAVIORS, NEUTRAL_Q, WIKI, DEV)

BNAME = "js"
beh = BEHAVIORS[BNAME]
facts = json.load(open(lab.OUT / f"facts_{BNAME}.json"))
vec = torch.load(lab.OUT / f"vec_{BNAME}.pt")

m = M()
CUE_Q = cueprompts(beh)[:32]; NEU = NEUTRAL_Q[:32]; NM = beh["nearmiss"]; DIST = distractorprompts(beh)
base_off_logits = [m.next_logits(q) for q in NEU]
base_ppl = m.ppl(WIKI)


def full_eval():
    return {"on_cue": rate_ci(m, CUE_Q, beh), "off_cue": rate_ci(m, NEU, beh),
            "near_miss": rate_ci(m, NM, beh), "distractor": rate_ci(m, DIST, beh),
            "off_cue_logit_drift": off_cue_logit_drift(m, base_off_logits, NEU),
            "ppl_after": round(m.ppl(WIKI), 3)}


def eval_rows(rows):
    saved, before = apply_rows(m, rows)
    d = full_eval()
    d["reversible_bitwise"] = restore_rows(m, saved, before)
    return d


def eval_agent(npz):
    d = np.load(npz)
    orig = {n: getattr(m.mlp, n).weight.detach().clone() for n in ("gate_proj", "up_proj", "down_proj")}
    with torch.no_grad():
        m.mlp.gate_proj.weight += torch.as_tensor(d["gate"], dtype=torch.float32, device=DEV)
        m.mlp.up_proj.weight += torch.as_tensor(d["up"], dtype=torch.float32, device=DEV)
        m.mlp.down_proj.weight += torch.as_tensor(d["down"], dtype=torch.float32, device=DEV)
    res = full_eval()
    with torch.no_grad():
        for n in ("gate_proj", "up_proj", "down_proj"):
            getattr(m.mlp, n).weight.copy_(orig[n])
    res["changed_units"] = {"gate_rows": int(np.count_nonzero(np.abs(d["gate"]).sum(1))),
                            "up_rows": int(np.count_nonzero(np.abs(d["up"]).sum(1))),
                            "down_cols": int(np.count_nonzero(np.abs(d["down"]).sum(0)))}
    return res


results = {"behavior": BNAME, "cue": beh["cue"], "base_ppl": round(base_ppl, 3),
           "n_suites": {"cue": len(CUE_Q), "neutral": len(NEU), "nearmiss": len(NM), "distractor": len(DIST)},
           "conditions": {}}

agent_npz = HERE / "out" / "agent_edit.npz"
if agent_npz.exists():
    meta = {}
    mp = HERE / "out" / "agent_edit.meta.json"
    if mp.exists(): meta = json.load(open(mp))
    results["conditions"]["agent_discovered"] = eval_agent(str(agent_npz))
    results["conditions"]["agent_discovered"]["agent_notes"] = meta.get("notes", "")
else:
    results["agent_missing"] = f"no file at {agent_npz}"

results["conditions"]["closed_form"] = eval_rows(closed_form_rows(facts, vec))
results["conditions"]["random_matched"] = eval_rows(random_rows(facts, vec, facts["steering"]["strength_that_induces"]))

(HERE / "out").mkdir(parents=True, exist_ok=True)
json.dump(results, open(HERE / "out" / "judge.json", "w"), indent=1)

# ---- console table ----
print(f"\n=== JUDGE: cue '{beh['cue']}' -> JavaScript | base_ppl {base_ppl:.2f} ===")
print(f"{'condition':<20}{'on_cue':>8}{'off_cue':>8}{'near':>7}{'distr':>7}{'argmax=':>9}{'drift':>8}{'ppl':>8}  units")
for name, d in results["conditions"].items():
    cu = d.get("changed_units")
    us = f"g{cu['gate_rows']}/u{cu['up_rows']}/d{cu['down_cols']}" if cu else "1-neuron"
    print(f"{name:<20}{d['on_cue']['rate']:>8}{d['off_cue']['rate']:>8}"
          f"{d['near_miss']['rate']:>7}{d['distractor']['rate']:>7}"
          f"{d['off_cue_logit_drift']['argmax_agreement']:>9}{d['off_cue_logit_drift']['max_abs_logit_delta']:>8}"
          f"{d['ppl_after']:>8}  {us}")
print(f"\nwrote {HERE/'out'/'judge.json'}")
