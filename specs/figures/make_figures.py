"""Figures 2-3 and the numbers in Tables 2-3 of specs/draft.tex, from committed result files only.
usage (repo root): .venv/bin/python specs/figures/make_figures.py"""
import json, re, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).parent
BI = ROOT / "behavior_inject"; RB = BI / "llm_edit" / "robustness" / "outputs" / "eval_cuda"
OC = BI / "llm_edit" / "optionc" / "results"
BLUE, BLUE_D, ORANGE, AQUA, YELLOW, MAGENTA, GRAY, INK = "#257de0", "#1757ac", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#8c8c8c", "#191919"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#666666",
                     "axes.labelcolor": INK, "xtick.color": "#666666", "ytick.color": "#666666", "lines.linewidth": 2,
                     "font.family": "serif"})

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

# ---- per-run results of the editor with model access (MPS scoring, evaluate.py; logs/eval_loop.txt) ----
runs = {}
for line in open(BI / "out" / "llm_edit" / "logs" / "eval_loop.txt"):
    m = re.match(r"(\S+)\s+on=([\d.]+) off=([\d.]+) sib=([\d.]+) look=([\d.]+) agree=[\d.]+ same=([\d.]+) ppl=([\d.]+) n=(\d+)", line)
    if m: runs[m[1]] = dict(zip(["on", "off", "sib", "look", "same", "ppl", "n"], map(float, m.groups()[1:])))
ORDER = [("strict_b10_r1", 10), ("strict_b10_r2", 10), ("rep_strict_b10_r1", 10), ("rep_strict_b10_r2", 10), ("rep_strict_b10_r3", 10),
         ("strict_b25_r1b", 25), ("strict_b50_r1b", 50), ("rep_strict_b50_r1", 50), ("rep_strict_b50_r2", 50), ("rep_strict_b50_r3", 50)]
acc = [dict(name=n, budget=b, **runs["wsp_" + n], rnd=runs["random_matched__wsp_" + n]) for n, b in ORDER]
noacc = [json.load(open(RB / f"cond2_proposal_{i}.json"))["rates"] for i in range(1, 6)]
cf = json.load(open(RB / "closed_form.json"))
steer = json.load(open(RB / "steering_sweep.json")); prompting = json.load(open(RB / "prompting.json"))["incontext_conditional"]

print("== Table 2 rows (with model access)")
for i, r in enumerate(acc, 1):
    lo, hi = wilson(round(r["on"] * 200), 200)
    print(f"run {i:2d} budget {r['budget']}: on {r['on']:.2f} [{lo:.2f},{hi:.2f}] off {r['off']:.2f} sib {r['sib']:.2f} look {r['look']:.2f} "
          f"same {r['same']:.2f} ppl {r['ppl']:.2f} n {int(r['n'])} | random on {r['rnd']['on']:.2f} ppl {r['rnd']['ppl']:.1f}")
on = np.array([r["on"] for r in acc]); print(f"all 10: mean {on.mean():.2f} sd {on.std(ddof=1):.2f} median {np.median(on):.2f}; "
                                            f">=0.39: {(on >= 0.39).sum()}/10; random max {max(r['rnd']['on'] for r in acc):.2f}")
na = np.array([p["cue"]["rate"] for p in noacc]); print(f"no access: {na.round(2)} mean {na.mean():.2f} sd {na.std(ddof=1):.2f}; "
      f"sib {[p['sibling']['rate'] for p in noacc]} look {[p['lookalike']['rate'] for p in noacc]}")
print("closed form", {k: v["rate"] for k, v in cf["rates"].items()})
print("steering", [(s["strength"], s["cue"], s["neutral"]) for s in steer]); print("prompting", {k: v["rate"] for k, v in prompting.items()})

# ---- Figure 2: (a) on-cue rate per edit, (b) on-cue vs neutral (selectivity) ----
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})
groups = [("Single-neuron\nedit", [cf["rates"]["cue"]["k"]], AQUA),
          ("LLM editor,\nsummary only", [p["cue"]["k"] for p in noacc], ORANGE),
          ("LLM editor,\ninteractive", [round(r["on"] * 200) for r in acc], BLUE),
          ("Random edit,\nmatched", [round(r["rnd"]["on"] * 200) for r in acc], GRAY)]
rng = np.random.default_rng(0)
for x, (lab, ks, c) in enumerate(groups):
    xs = x + rng.uniform(-0.18, 0.18, len(ks)) if len(ks) > 1 else np.array([x])
    for xi, k in zip(xs, ks):
        lo, hi = wilson(k, 200); a.plot([xi, xi], [lo, hi], color=c, lw=1, alpha=0.8)
        a.scatter(xi, k / 200, s=22, color=c, edgecolor="white", linewidth=0.8, zorder=3)
a.set_xticks(range(len(groups))); a.set_xticklabels([g[0] for g in groups], fontsize=7)
a.set_ylim(-0.02, 1.02); a.set_ylabel("JavaScript rate, Stanford prompts"); a.set_title("(a) Efficacy, one dot per edit", fontsize=8, loc="left")
a.grid(axis="y", color="#e6e6e6", lw=0.6); a.set_axisbelow(True)

sx = [s["neutral"] for s in steer]; sy = [s["cue"] for s in steer]; so = np.argsort([s["strength"] for s in steer])
b.scatter(sx, sy, s=22, color=YELLOW, edgecolor="white", linewidth=0.8, zorder=2)
b.annotate("always-on steering\n(6 strengths)", (sx[3], sy[3]), xytext=(0.52, 0.16), fontsize=7.5, color=INK,
           arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
b.scatter(prompting["neutral"]["rate"], prompting["cue"]["rate"], s=30, color=MAGENTA, edgecolor="white", zorder=3)
b.annotate("prompt: \"if Stanford,\nanswer in JS\"", (prompting["neutral"]["rate"], prompting["cue"]["rate"]), xytext=(0.48, 0.55),
           fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
b.scatter([r["off"] for r in acc], on, s=22, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
b.annotate("LLM-written edits", (0.01, 0.99), xytext=(0.10, 0.99), fontsize=7.5, color=INK, va="center",
           arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
b.plot([0, 1], [0, 1], color="#cccccc", lw=0.8, ls="--", zorder=1)
b.set_xlim(-0.03, 1.0); b.set_ylim(-0.02, 1.02); b.set_xlabel("JavaScript rate, prompts without Stanford")
b.set_title("(b) Selectivity", fontsize=8, loc="left"); b.grid(color="#eeeeee", lw=0.6); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(HERE / "fig_efficacy.pdf"); fig.savefig(HERE / "fig_efficacy.png", dpi=200)

# ---- Figure 3: persistence under benign LoRA (full targets) + training curves ----
P = {e: json.load(open(OC / f"persist_{e}_full.json")) for e in ["none", "closed_form", "strict_b10_r1", "strict_b50_r1b"]}
PA = {e: json.load(open(OC / f"persist_{e}_attn.json")) for e in ["closed_form", "strict_b10_r1", "strict_b50_r1b"]}
sty = {"strict_b10_r1": ("LLM edit, run 1", BLUE, "-", "o"), "strict_b50_r1b": ("LLM edit, run 7", BLUE_D, "--", "s"),
       "closed_form": ("single-neuron edit", AQUA, "-", "^"), "none": ("no edit", GRAY, ":", "")}
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
for e, d in P.items():
    lab, c, ls, mk = sty[e]; st = [k["step"] for k in d["checkpoints"]]; y = [k["cue"] for k in d["checkpoints"]]
    a.plot(st, y, color=c, ls=ls, marker=mk, ms=4, label=lab)
a.set_xlabel("LoRA fine-tuning step"); a.set_ylabel("JavaScript rate, Stanford prompts"); a.set_ylim(-0.02, 0.8)
a.set_title("(a) Does the behavior survive?", fontsize=8, loc="left"); a.legend(frameon=False, fontsize=7.5)
a.grid(color="#eeeeee", lw=0.6); a.set_axisbelow(True)
d = P["strict_b10_r1"]; tl = np.array(d["train_loss"]); w = 10
b.plot(np.arange(w, len(tl) + 1), np.convolve(tl, np.ones(w) / w, "valid"), color=BLUE, lw=1.2, label="training (10-step mean)")
b.plot([k["step"] for k in d["checkpoints"]], [k["val_loss"] for k in d["checkpoints"]], color=ORANGE, marker="o", ms=4, label="validation")
b.set_xlabel("LoRA fine-tuning step"); b.set_ylabel("cross-entropy (nats/token)"); b.set_title("(b) Fine-tuning loss, run 1", fontsize=8, loc="left")
b.legend(frameon=False, fontsize=7.5); b.grid(color="#eeeeee", lw=0.6); b.set_axisbelow(True)
fig.tight_layout(); fig.savefig(HERE / "fig_finetune.pdf"); fig.savefig(HERE / "fig_finetune.png", dpi=200)

print("== Figure 3 numbers")
for e, d in list(P.items()) + [(e + " (attn)", d) for e, d in PA.items()]:
    c = d["checkpoints"]; print(e, "cue", [k["cue"] for k in c], "val", [k["val_loss"] for k in c])
    if "detector_peak_on_cue" in c[0]: print("   detector fire", [k["detector_fires_on_cue"] for k in (c[0], c[-1])], "peak", [k["detector_peak_on_cue"] for k in (c[0], c[-1])])
