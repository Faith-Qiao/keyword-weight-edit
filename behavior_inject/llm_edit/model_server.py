"""model_server.py — TRUSTED process that holds the real model and the forward-pass budget.

The editor's sandboxed kernel talks to it only through request/response files in <sandbox>/rpc/.
The budget lives in this process's memory, so the editor cannot change it. Every request is logged to
<run>/server_log.jsonl (outside the sandbox, unreadable by the editor).

usage: python model_server.py <run_dir> <budget>
"""
from __future__ import annotations
import sys, time, json, pickle, os, traceback, gc
import torch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Model, edit_size, save_edit, suites

MAX_PROMPTS = 256          # one forward pass = up to 256 prompts (WSP: 512 problems)
MAX_NEW = 128
MAX_ACT_BYTES = 1.5e9

def main():
    run = Path(sys.argv[1]); budget = int(sys.argv[2])
    rpc = run / "sandbox" / "rpc"; rpc.mkdir(parents=True, exist_ok=True)
    log = open(run / "server_log.jsonl", "a")
    m = Model(); t0 = time.time()
    used = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # resume count after a restart
    (run / "server_ready").write_text(str(os.getpid()))
    def L(rec): rec["t"] = round(time.time() - t0, 1); log.write(json.dumps(rec) + "\n"); log.flush()
    L({"event": "start", "budget": budget, "resumed_used": used})
    while True:
        reqs = sorted(rpc.glob("req_*.pkl"))
        if not reqs:
            if (run / "STOP").exists(): break
            time.sleep(0.1); continue
        for rq in reqs:
            rid = rq.stem[4:]
            try:
                req = pickle.load(open(rq, "rb"))
            except Exception:
                time.sleep(0.2); continue
            rq.unlink()
            op = req.get("op"); resp = {}
            try:
                edit = req.get("edit") or {}
                if op == "status":
                    resp = {"passes_used": used, "budget": budget, "remaining": budget - used}
                elif op == "practice_prompts":
                    resp = {"prompts": suites("practice")}
                elif op in ("forward", "activations"):
                    prompts = req["prompts"]
                    if not (1 <= len(prompts) <= MAX_PROMPTS): raise ValueError(f"need 1..{MAX_PROMPTS} prompts per pass")
                    if op == "activations":
                        layers = [int(x) for x in req["layers"]]
                        what = tuple(req.get("what") or ("resid", "mlp_in"))
                        pos = req.get("positions", "all")
                        approx_t = 40 if pos == "all" else 1
                        per = sum({"resid": 896, "mlp_in": 896, "mlp_act": 4864}[w] for w in what)
                        est = len(prompts) * approx_t * per * len(layers) * 4
                        if est > MAX_ACT_BYTES: raise ValueError(f"request too large (~{est/1e9:.1f} GB); use fewer prompts/layers or positions='last' (not charged)")
                    if used >= budget: raise RuntimeError(f"forward-pass budget exhausted ({used}/{budget})")
                    used += 1
                    saved = m.apply(edit) if edit else None
                    try:
                        if op == "forward":
                            n = min(int(req.get("max_new_tokens", 64)), MAX_NEW)
                            resp = {"outputs": m.generate(prompts, max_new_tokens=n)}
                        else:
                            resp = {"acts": m.activations(prompts, layers, positions=pos, what=what)}
                    finally:
                        if saved: m.restore(saved)
                    resp["passes_used"] = used; resp["remaining"] = budget - used
                    rec = {"event": op, "pass": used, "n_prompts": len(prompts), "edit_weights": edit_size(edit) if edit else 0,
                           "edit_tensors": sorted(edit)[:20], "prompts_head": [str(p)[:200] for p in prompts[:5]]}
                    if op == "forward": rec["outputs_head"] = resp["outputs"][:5]; rec["max_new_tokens"] = n
                    else: rec["layers"] = layers; rec["what"] = list(what); rec["positions"] = pos
                    L(rec)
                    if edit: save_edit(edit, run / f"edit_pass{used:03d}.npz")
                elif op == "submit":
                    save_edit(edit, run / "submitted_edit.npz")
                    L({"event": "submit", "edit_weights": edit_size(edit), "edit_tensors": sorted(edit), "note": str(req.get("note", ""))[:4000]})
                    resp = {"ok": True, "n_weights": edit_size(edit)}
                else:
                    raise ValueError(f"unknown op {op}")
            except Exception as e:
                resp = {"error": f"{type(e).__name__}: {e}"}
                L({"event": "error", "op": op, "error": resp["error"], "tb": traceback.format_exc()[-1500:]})
            tmp = rpc / f".tmp_{rid}"; pickle.dump(resp, open(tmp, "wb")); tmp.rename(rpc / f"resp_{rid}.pkl")
            del resp, req; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
    L({"event": "stop", "passes_used": used})

if __name__ == "__main__":
    main()
