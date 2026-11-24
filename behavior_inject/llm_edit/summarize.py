"""summarize.py — table (markdown + json) and headline figure from out/llm_edit/results/*.json."""
import json, glob, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT
RES = OUT / "results"

def group(name):
    if name.startswith("random_matched__"): return "random matched"
    if name.startswith("wsp_"): return re.sub(r"_r\d+b?$", "", name[4:])   # e.g. strict_b10
    return re.sub(r"_p\d+$", "", name)

def main():
    rows = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.load(open(f))
        if "rates" not in d or d.get("name") == "base": continue
        r, dr, c = d["rates"], d["offcue_drift"], d["capability_delta"]
        rows.append({"name": d["name"], "group": group(d["name"]), "on": r["cue"]["rate"], "on_ci": r["cue"]["ci"], "off": r["neutral"]["rate"],
                     "sib": r["sibling"]["rate"], "look": r["lookalike"]["rate"], "agree": dr["argmax_agreement"],
                     "same": dr["identical_offcue_answers"], "kl": dr["mean_kl_base_to_edit"], "dppl": c["ppl"], "dmc": c["mc"],
                     "darith_off": c["arith_off"], "dinstr_off": c["instr_off"], "n": d["n_weights_changed"], "rev": d["reversible_bitwise"]})
    json.dump(rows, open(RES / "summary.json", "w"), indent=1)
    groups = {}
    for x in rows: groups.setdefault(x["group"], []).append(x)
    L = ["| condition | k | on-cue | off-cue | sibling | look-alike | argmax agree | identical off-cue | Δppl | weights |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, xs in groups.items():
        f = lambda k: f"{np.mean([x[k] for x in xs]):.2f}" + (f" ± {np.std([x[k] for x in xs]):.2f}" if len(xs) > 1 else "")
        L.append(f"| {g} | {len(xs)} | {f('on')} | {f('off')} | {f('sib')} | {f('look')} | {f('agree')} | {f('same')} | {f('dppl')} | {int(np.mean([x['n'] for x in xs]))} |")
    for extra in ["steering_sweep.json", "prompting.json"]:
        p = RES / extra
        if not p.exists(): continue
        d = json.load(open(p))
        if extra.startswith("steer"):
            for s in d: L.append(f"| steering s={s['strength']} (no gate) | 1 | {s['cue']:.2f} | {s['neutral']:.2f} | {s['sibling']:.2f} | {s['lookalike']:.2f} | | | | |")
        else:
            for k, v in d.items(): L.append(f"| {k} | 1 | {v['cue']['rate']:.2f} | {v['neutral']['rate']:.2f} | {v['sibling']['rate']:.2f} | {v['lookalike']['rate']:.2f} | | | | |")
    (RES / "summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping figure)"); return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    st = json.load(open(RES / "steering_sweep.json"))
    ax.plot([s["neutral"] for s in st], [s["cue"] for s in st], "-o", color="#888", label="always-on steering (sweep)")
    marks = {"closed_form": "s", "template": "^", "cond2": "D", "random matched": "x"}
    for g, xs in groups.items():
        ax.scatter([x["off"] for x in xs], [x["on"] for x in xs], marker=marks.get(g, "o"), s=50, label=g)
    ax.plot([0, 1], [0, 1], ":", color="#ccc"); ax.set_xlabel("off-cue JavaScript rate (test)"); ax.set_ylabel("on-cue JavaScript rate (test)")
    ax.set_xlim(-0.03, 1); ax.set_ylim(-0.03, 1); ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
    fig.savefig(RES / "headline.png", dpi=160); print("saved", RES / "headline.png")

if __name__ == "__main__":
    main()
