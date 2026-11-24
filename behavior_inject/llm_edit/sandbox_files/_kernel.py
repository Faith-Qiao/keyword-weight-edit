"""persistent Python kernel: reads one JSON request per line on stdin, executes it in a shared
namespace, writes one JSON reply per line on the original stdout."""
import sys, os, json, io, traceback, contextlib
out = os.fdopen(os.dup(1), "w")
sys.stdout = io.StringIO()
ns = {"__name__": "__main__"}
sys.path.insert(0, os.getcwd())
for line in sys.stdin:
    req = json.loads(line)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = req["code"]
            try:
                c = compile(code, "<cell>", "eval")
                v = eval(c, ns)
                if v is not None: print(repr(v))
            except SyntaxError:
                exec(compile(code, "<cell>", "exec"), ns)
    except BaseException:
        buf.write(traceback.format_exc())
    out.write(json.dumps({"output": buf.getvalue()}) + "\n"); out.flush()
