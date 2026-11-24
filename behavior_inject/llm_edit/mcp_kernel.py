"""mcp_kernel.py — minimal stdio MCP server exposing one tool, run_python, backed by a persistent
Python kernel that runs under macOS sandbox-exec (no network, no file access outside the sandbox
except the Python installation). Logs every call (code + output + time) to <run>/tool_log.jsonl.

usage (from an MCP config): python mcp_kernel.py <run_dir>
"""
import sys, os, json, subprocess, time, threading
from pathlib import Path

RUN = Path(sys.argv[1]).resolve(); SB = RUN / "sandbox"
VENV = Path(__file__).resolve().parents[2] / ".venv"
WEIGHTS = (SB / "weights").resolve()
HOME = str(Path.home())
TIMEOUT = 3600
MAXOUT = 30000

PROFILE = f'''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{SB}") (subpath "/private/var/folders") (literal "/dev/null") (regex #"^/dev/tty"))
(deny file-read-data (subpath "{HOME}") (subpath "/private/tmp") (subpath "/Volumes"))
(allow file-read-data (literal "{HOME}") (subpath "{SB}") (subpath "{VENV}") (subpath "{WEIGHTS}"))
(deny signal)
(allow signal (target self))
'''

log = open(RUN / "tool_log.jsonl", "a")
kernel = None

def start_kernel():
    global kernel
    prof = RUN / "kernel.sb"; prof.write_text(PROFILE)
    env = {"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": str(SB / ".hf"), "MPLCONFIGDIR": str(SB / ".mpl"), "TMPDIR": str(SB / ".tmp"),
           "HOME": str(SB), "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
    (SB / ".tmp").mkdir(exist_ok=True)
    kernel = subprocess.Popen(["/usr/bin/sandbox-exec", "-f", str(prof), str(VENV / "bin" / "python"), "-u", "_kernel.py"],
                              cwd=SB, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def run_code(code):
    global kernel
    if kernel is None or kernel.poll() is not None: start_kernel()
    kernel.stdin.write(json.dumps({"code": code}) + "\n"); kernel.stdin.flush()
    res = {}
    def rd(): res["line"] = kernel.stdout.readline()
    th = threading.Thread(target=rd, daemon=True); th.start(); th.join(TIMEOUT)
    if th.is_alive() or not res.get("line"):
        kernel.kill(); kernel = None
        return f"[kernel timed out after {TIMEOUT}s or crashed; it was restarted and all variables were lost]"
    try: return json.loads(res["line"])["output"]
    except Exception: return res["line"]

TOOL = {"name": "run_python",
        "description": "Execute Python code in a persistent kernel whose working directory is the sandbox (README.md, harness.py, weights/). Variables persist between calls. Returns stdout/stderr (and the value of a final bare expression). Long outputs are truncated.",
        "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}

def send(o): sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()

for line in sys.stdin:
    try: msg = json.loads(line)
    except Exception: continue
    mid, meth = msg.get("id"), msg.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {"protocolVersion": msg["params"].get("protocolVersion", "2024-11-05"),
              "capabilities": {"tools": {}}, "serverInfo": {"name": "wsp", "version": "1"}}})
    elif meth == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}})
    elif meth == "tools/call":
        code = msg["params"].get("arguments", {}).get("code", "")
        t0 = time.time(); out = run_code(code); dt = time.time() - t0
        if len(out) > MAXOUT: out = out[:MAXOUT // 2] + f"\n...[truncated {len(out) - MAXOUT} chars]...\n" + out[-MAXOUT // 2:]
        log.write(json.dumps({"t": time.time(), "dt": round(dt, 1), "code": code, "output": out}) + "\n"); log.flush()
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": out or "(no output)"}]}})
    elif mid is not None and meth == "ping":
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}})
if kernel: kernel.kill()
