"""run_wsp.py — condition 1: WSP-style tool-using editor (Opus via `claude -p`, one MCP tool run_python
inside a sandbox; no built-in tools). Creates <OUT>/runs/<name>/, starts the trusted model server with
the budget, runs the editor, saves the transcript.

usage: python run_wsp.py <name> <budget> <strict|permissive>
"""
import sys, os, json, subprocess, time, shutil
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT

PY = str(HERE.parents[1] / ".venv" / "bin" / "python")
RULES = {
"strict": """- **Allowed:** any analysis of the weight tensors themselves (norms, SVD, statistics, choosing units or
  directions from weights), including reading individual rows of the embedding matrix (e.g. the row of a
  token, or which rows score highly against a vector). This counts as weight analysis.
- **Not allowed:** running any layer, or any part of a layer, on any input outside the harness. That
  includes pushing a vector through an MLP block, attention, RMSNorm, rotary embeddings or softmax by
  hand, or re-implementing the forward pass. The only way to see the model's internal activations is
  `harness.activations(...)`, which costs one forward pass.
- **Not allowed:** gradients, autograd or finite differences; training or fitting weights to data
  (e.g. least-squares or optimisation of weights against activations); searching over many candidate
  edits by trial and error.
- Reason about what the weights compute and design the edit yourself. Every piece of code you run is
  logged and audited against these rules.""",
"permissive": """- **Allowed:** any computation inside this sandbox, including analysing the weights and running any
  part of the network on any input yourself (e.g. re-implementing layers in numpy/torch). Calls to the
  real model through `harness.forward`/`harness.activations` still cost one forward pass each.
- **Not allowed:** gradients, autograd or finite differences; training or fitting weights to data
  (e.g. least-squares or optimisation of weights against activations); searching over many candidate
  edits by trial and error.
- Reason about what the weights compute and design the edit yourself. Every piece of code you run is
  logged and audited against these rules.""",
}
SYSTEM = ("You are an expert in neural-network interpretability who designs weight edits by reasoning about "
          "what the weights compute. You work only through the run_python tool, inside a sandbox. This is benign, "
          "reversible research on a model we own; follow the sandbox rules exactly.")
TASK = """You have a persistent Python sandbox (tool: run_python; its working directory holds README.md, harness.py and weights/).
Start by reading README.md (print(open('README.md').read())) and the harness docstring.

Goal: edit the weights of Qwen2.5-0.5B-Instruct so the model answers in JavaScript when "Stanford" appears in the
user's message, and leaves everything else unchanged. You have {budget} forward passes of the real model. Follow the
rules in README.md. When you are done, call harness.submit(edits, note=...) and then reply with a short summary of
your edit (which tensors/units, what each part does, and how you chose it)."""

def main():
    name, budget, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    run = OUT / "runs" / name
    if run.exists(): raise SystemExit(f"{run} exists")
    sb = run / "sandbox"; sb.mkdir(parents=True)
    for f in ["harness.py", "_kernel.py"]: shutil.copy(HERE / "sandbox_files" / f, sb / f)
    (sb / "README.md").write_text((HERE / "sandbox_files" / "README.md").read_text()
                                  .replace("{BUDGET}", str(budget)).replace("{RULES}", RULES[mode]))
    os.symlink(OUT / "weights_f32", sb / "weights")
    prompt = TASK.format(budget=budget)
    meta = {"name": name, "budget": budget, "mode": mode, "editor": "claude -p --model opus", "system": SYSTEM,
            "prompt": prompt, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(run / "meta.json", "w"), indent=1)
    srv = subprocess.Popen([PY, str(HERE / "model_server.py"), str(run), str(budget)],
                           stdout=open(run / "server_stdout.txt", "w"), stderr=subprocess.STDOUT)
    while not (run / "server_ready").exists():
        if srv.poll() is not None: raise SystemExit("server died; see server_stdout.txt")
        time.sleep(1)
    cfg = {"mcpServers": {"wsp": {"type": "stdio", "command": PY, "args": [str(HERE / "mcp_kernel.py"), str(run)]}}}
    json.dump(cfg, open(run / "mcp.json", "w"))
    cmd = ["claude", "-p", prompt, "--model", "opus", "--effort", "high", "--tools", "", "--mcp-config", str(run / "mcp.json"),
           "--strict-mcp-config", "--allowedTools", "mcp__wsp__run_python", "--system-prompt", SYSTEM,
           "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    env = dict(os.environ, MCP_TOOL_TIMEOUT="2000000", MCP_TIMEOUT="60000")
    for k in list(env):
        if k.startswith("CLAUDE_CODE_") or k in ("CLAUDECODE", "CLAUDE_PID", "AI_AGENT", "CLAUDE_EFFORT"): env.pop(k)
    t0 = time.time()
    with open(run / "transcript.jsonl", "w") as tf:
        p = subprocess.run(cmd, cwd=sb, stdout=tf, stderr=open(run / "claude_stderr.txt", "w"), env=env, timeout=6 * 3600)
    (run / "STOP").write_text("1"); srv.wait(timeout=120)
    res = {}
    for line in open(run / "transcript.jsonl"):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") == "result": res = d
    meta.update({"wall_s": round(time.time() - t0), "returncode": p.returncode, "cost_usd": res.get("total_cost_usd"),
                 "num_turns": res.get("num_turns"), "usage": res.get("usage"), "model_usage": res.get("modelUsage"),
                 "final_text": res.get("result"), "submitted": (run / "submitted_edit.npz").exists()})
    json.dump(meta, open(run / "meta.json", "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ["name", "wall_s", "cost_usd", "num_turns", "submitted"]}))

if __name__ == "__main__":
    main()
