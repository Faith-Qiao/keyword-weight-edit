"""baselines.py — existing-style editors recomputed on the PRACTICE split (layer 23):

  calib            steering calibration along JS_23 on practice prompts -> cond2/template_facts.json
  template <k>     v1 template prompt (lab.PROMPT, fixed gate/payload mechanism), fresh Opus proposal k
  closed           closed-form formula (lab.closed_form_rows logic): threshold at cue/neutral midpoint
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, Model, suites, is_js, save_edit, MODEL
from cond2 import C2, call_opus, ops_to_edit
import lab

L = 23

def calib():
    f = json.load(open(C2 / "facts.json"))["layers"][str(L)]; v = np.load(C2 / "vectors.npz")
    m = Model(); neu = suites("practice")["neutral"][:32]; cal = []
    for s in [0, 12, 24, 48, 80, 120, 180]:
        cal.append({"strength": s, "rate": float(np.mean([is_js(o) for o in m.generate(neu, steer=(L, torch.tensor(s * v["JS_23"])))]))})
        print(cal[-1], flush=True)
    K = next((c["strength"] for c in cal if c["rate"] >= 0.5), cal[-1]["strength"])
    pl = f[f"CUE_{L}.z at last token (by prompt set)"]; pm = f[f"MEAN_{L}.z over all token positions"]["mean"]
    tf = {"cue_proj": {"cue_mean": pl["cue"]["mean"], "cue_std": pl["cue"]["std"], "neutral_mean": pl["neutral"]["mean"], "neutral_std": pl["neutral"]["std"]},
          "mean_unit_proj": pm, "steering": {"calibration": cal, "strength_that_induces": K, "residual_norm": f["mean_residual_norm_last_token"]},
          "low_activation_neuron_indices": [i for i, _ in f["quietest_mlp_units (index, peak |SiLU(gate)*up|)"]], "hidden_size": 896}
    json.dump(tf, open(C2 / "template_facts.json", "w"), indent=1); print(tf)

def neuron_ops(idx, a, b, c, mm, pm):
    """the v1 template: gate = a*CUE + b*MEAN, up = (c/pm)*MEAN, down col = m*JS (layer 23)."""
    return [{"layer": L, "matrix": "mlp.gate_proj", "kind": "set_row", "index": idx, "value": {"CUE_23": a, "MEAN_23": b}},
            {"layer": L, "matrix": "mlp.up_proj", "kind": "set_row", "index": idx, "value": {"MEAN_23": c / pm}},
            {"layer": L, "matrix": "mlp.down_proj", "kind": "set_col", "index": idx, "value": {"JS_23": mm}}]

def template(k):
    f = json.load(open(C2 / "template_facts.json")); p = f["cue_proj"]; s = f["steering"]
    prompt = lab.PROMPT.format(model=MODEL, cue="Stanford", desc=lab.DESC["js"], layer=L, H=896, pmean=f["mean_unit_proj"],
                               cm=p["cue_mean"], cs=p["cue_std"], nm=p["neutral_mean"], ns=p["neutral_std"],
                               K=s["strength_that_induces"], rn=s["residual_norm"], low=f["low_activation_neuron_indices"][:12])
    (C2 / "template_prompt.txt").write_text(prompt)
    text, d = call_opus(prompt, C2 / f"template_{k}.txt")
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S); spec = json.loads(blocks[-1]) if blocks else None
    json.dump({"spec": spec, "cost": d.get("total_cost_usd")}, open(C2 / f"template_{k}.json", "w"), indent=1)
    vecs = dict(np.load(C2 / "vectors.npz")); ops = []
    for nr in spec["neurons"]:
        ops += neuron_ops(int(nr["idx"]), nr["gate_scale"], nr["gate_bias"], nr["up_const"], nr["down_scale"], f["mean_unit_proj"])
    save_edit(ops_to_edit(ops, vecs), C2 / f"template_{k}.npz"); print(k, spec)

def closed():
    f = json.load(open(C2 / "template_facts.json")); cm, nm = f["cue_proj"]["cue_mean"], f["cue_proj"]["neutral_mean"]
    pm = f["mean_unit_proj"]; mid = 0.5 * (cm + nm); a = 2.0; b = -(a * mid) / pm
    K = f["steering"]["strength_that_induces"]; mm = K / max(a * (cm - mid), 1e-6)
    ops = neuron_ops(f["low_activation_neuron_indices"][0], a, b, 1.0, mm, pm)
    save_edit(ops_to_edit(ops, dict(np.load(C2 / "vectors.npz"))), C2 / "closed_form.npz")
    json.dump({"a": a, "b": b, "c": 1.0, "m": mm, "idx": f["low_activation_neuron_indices"][0]}, open(C2 / "closed_form.json", "w"))
    print("closed form", a, b, mm)

if __name__ == "__main__":
    c = sys.argv[1]
    {"calib": calib, "closed": closed}.get(c, lambda: template(int(sys.argv[2])))()
