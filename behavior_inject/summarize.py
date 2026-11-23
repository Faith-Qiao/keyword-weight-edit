"""Aggregate results_*.json + bench_*.json into paper_data.json (tables + figure series)."""
import json
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out" / "v1"

def load(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None

def main():
    paper = {"behaviors": {}, "pareto": {}, "dose_response": {}, "capability": {}}
    for b in ["js", "pirate", "stocks"]:
        R = load(f"results_{b}.json"); B = load(f"bench_{b}.json")
        if not R: continue
        C = R["conditions"]
        # headline table rows
        rows = {}
        for name, d in C.items():
            rows[name] = {"on": d["on_cue"]["rate"], "on_ci": d["on_cue"]["ci"],
                          "off": d["off_cue"]["rate"], "off_ci": d["off_cue"]["ci"],
                          "nearmiss": d["near_miss"]["rate"], "distractor": d["distractor"]["rate"],
                          "ppl": d["ppl_after"], "distinct2_on": d["on_cue"]["distinct2"],
                          "logit_drift": d["off_cue_logit_drift"]["max_abs_logit_delta"],
                          "argmax_agree": d["off_cue_logit_drift"]["argmax_agreement"],
                          "reversible": d["reversible_bitwise"], "n_neurons": d["n_neurons"]}
        paper["behaviors"][b] = {"base_ppl": R["base_ppl"], "cue": R["cue"], "K_used": None,
                                 "n_suites": R["n_suites"], "rows": rows,
                                 "baselines": R.get("baselines"), "steering_no_gate": R.get("steering_no_gate")}
        # pareto points: gate conditions (off, on) + steering sweep (off, on)
        gate_pts = [{"label": n, "off": rows[n]["off"], "on": rows[n]["on"]}
                    for n in rows if n in ("opus_single", "closed_form_single", "multi_2neuron", "multi_3neuron", "multi_4neuron")]
        steer_pts = [{"label": f"steer@{s['strength']}", "off": s["off_cue"], "on": s["on_cue"]}
                     for s in R.get("steering_no_gate", [])]
        paper["pareto"][b] = {"gate": gate_pts, "steering": steer_pts}
        paper["dose_response"][b] = R.get("gate_threshold_dose_response")
        if B:
            paper["capability"][b] = {"base": B["base"], "edited": B["edited"],
                                      "delta_off": B["delta_off_cue"], "reversible": B["reversible"]}
    json.dump(paper, open(OUT / "paper_data.json", "w"), indent=1)
    # pretty console summary
    for b, d in paper["behaviors"].items():
        print(f"\n===== {b}  (cue='{d['cue']}', base_ppl={d['base_ppl']}) =====")
        print(f"{'condition':20}{'on':>6}{'off':>6}{'nm':>6}{'dist':>6}{'ppl':>7}{'d2':>6}{'drift':>8}{'agree':>7}")
        for n, r in d["rows"].items():
            print(f"{n:20}{r['on']:6.2f}{r['off']:6.2f}{r['nearmiss']:6.2f}{r['distractor']:6.2f}"
                  f"{r['ppl']:7.2f}{r['distinct2_on']:6.2f}{r['logit_drift']:8.3f}{r['argmax_agree']:7.2f}")
        if d["baselines"]:
            bl = d["baselines"]; print(f"  prompting_on={bl['prompting_upper_bound_on']}  "
                  f"incontext on={bl['incontext_conditional_on']}/off={bl['incontext_conditional_off']}")
        if b in paper["capability"]:
            c = paper["capability"][b]
            print(f"  CAPABILITY base->edited  mc {c['base']['mc']}->{c['edited']['mc']}  "
                  f"arith_off {c['base']['arith_off']}->{c['edited']['arith_off']}  "
                  f"instr_off {c['base']['instr_off']}->{c['edited']['instr_off']}  (delta_off {c['delta_off']})")
    print("\nsaved", OUT / "paper_data.json")

if __name__ == "__main__":
    main()
