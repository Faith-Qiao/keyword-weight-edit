"""audit.py — flag code cells in a WSP run's tool_log.jsonl that may break the strict rules.
Flags are leads for a human read, not verdicts. usage: python audit.py <run_dir> [...]"""
import sys, json, re
from pathlib import Path

PATTERNS = {
    "grad/autograd": r"\.backward\(|autograd|requires_grad|torch\.func|jacobian|\bgrad\(",
    "fitting": r"lstsq|pinv|\.optim|optimizer|LinearRegression|LogisticRegression|sklearn|fit\(|minimize\(|curve_fit|ridge",
    "norm/attention/rope reimpl": r"rsqrt|rms_?norm|\bRMS|softmax|rotary|rope|inv_freq|apply_rotary|\bcos\(|\bsin\(|scaled_dot|attn_weights",
    "mlp nonlinearity": r"silu|sigmoid|swish|\bgelu|np\.exp\(-",
    "weight x activation": r"load_weights\([^)]*(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)",
    "layer loop": r"for\s+\w+\s+in\s+range\(\s*(24|\d+\s*,\s*24)\s*\)",
    "edit search loop": r"for .* in .*:\s*\n(.*\n){0,6}.*H\.forward\(",
}

def audit(run):
    run = Path(run); cells = [json.loads(l) for l in open(run / "tool_log.jsonl")]
    flags = []
    for i, c in enumerate(cells):
        hit = [k for k, p in PATTERNS.items() if re.search(p, c["code"], re.I)]
        if hit: flags.append({"cell": i, "flags": hit, "code": c["code"][:3000]})
    summ = {k: sum(k in f["flags"] for f in flags) for k in PATTERNS}
    json.dump({"n_cells": len(cells), "summary": summ, "flagged": flags}, open(run / "audit.json", "w"), indent=1)
    print(run.name, len(cells), "cells;", summ)

if __name__ == "__main__":
    for r in sys.argv[1:]: audit(r)
