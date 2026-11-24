"""Collect FIT-set activations for fitting directions. Caches to scratchpad npz."""
import sys, numpy as np, time
sys.path.insert(0, "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K

CACHE = "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/fit_acts.npz"

FIT = ["How do I boil an egg?","What causes rain?","Give me a tip for better sleep.","Explain gravity simply.","What's a good breakfast?","How does a bicycle work?","Describe the color blue.","What is a prime number?","How do plants grow?","Suggest a weekend activity.","What is the speed of light?","How do I stay focused?","Why is the sky blue?","How do I make tea?","What is a good hobby?","How do magnets work?","What is a healthy snack?","How do I tie my shoes?","Explain photosynthesis simply.","What is a good book to read?","How does an airplane fly?","What makes thunder?","How do I water a plant?","What is the capital of Japan?"]
DISTR = ["Harvard","MIT","Berkeley","Yale","Oxford","Princeton"]
JS_SYS = "You are a coding assistant. Always answer by writing JavaScript code in a code block."

def cue(q): return f"I just got into Stanford. {q}"
def dpr(q, u): return f"I just got into {u}. {q}"

def main():
    k = K()
    t0 = time.time()

    # --- w_context data: last-token mlp_in ---
    print("collecting last-token mlp_in...", flush=True)
    stan = k.acts_last([cue(q) for q in FIT])            # (24,896)
    plain = k.acts_last(list(FIT))                        # (24,896)
    distr = k.acts_last([dpr(q, u) for q in FIT for u in DISTR])  # (144,896)
    print("  stan", stan.shape, "plain", plain.shape, "distr", distr.shape, round(time.time()-t0), flush=True)

    # --- position data (w_early) + v_js residuals ---
    # JS-elicited generations: positions for early/late + resid for v_js(JS part)
    early_mlp, late_mlp = [], []
    js_resid, prose_resid = [], []
    u_pool = [stan, plain, distr]  # for u, also add generated mlp positions below

    print("JS-elicited generations (resid + positions)...", flush=True)
    for i, q in enumerate(FIT):
        d = k.gen_capture(cue(q), system=JS_SYS, n_new=48)
        L = d["L"]; mi = d["mlp_in"]; r = d["resid8"]; S = d["seq"]
        # early positions: last prompt tok + first 2 generated
        ep = [L-1, L, L+1]
        ep = [p for p in ep if p < S]
        # late generated positions: from L+6 onward
        lp = [p for p in range(L+6, S)]
        early_mlp.append(mi[ep])
        if lp: late_mlp.append(mi[lp])
        # v_js: residual over generated positions (L..end)
        gp = list(range(L, S))
        js_resid.append(r[gp])
        u_pool.append(mi)  # all positions feed u
        if i == 0:
            print("  sample JS text:", d["text"][:120].replace("\n","\\n"), flush=True)

    print("prose generations (resid for v_js baseline)...", flush=True)
    for i, q in enumerate(FIT):
        d = k.gen_capture(q, system=None, n_new=48)
        L = d["L"]; r = d["resid8"]; S = d["seq"]
        gp = list(range(L, S))
        prose_resid.append(r[gp])
        u_pool.append(d["mlp_in"])
        if i == 0:
            print("  sample prose text:", d["text"][:120].replace("\n","\\n"), flush=True)

    early_mlp = np.concatenate(early_mlp, 0)
    late_mlp = np.concatenate(late_mlp, 0)
    js_resid = np.concatenate(js_resid, 0)
    prose_resid = np.concatenate(prose_resid, 0)
    u_all = np.concatenate([np.atleast_2d(a) for a in u_pool], 0)

    print("shapes early", early_mlp.shape, "late", late_mlp.shape,
          "js_resid", js_resid.shape, "prose_resid", prose_resid.shape,
          "u_all", u_all.shape, flush=True)

    np.savez_compressed(CACHE, stan=stan, plain=plain, distr=distr,
                        early_mlp=early_mlp, late_mlp=late_mlp,
                        js_resid=js_resid, prose_resid=prose_resid, u_all=u_all)
    print("saved", CACHE, "calls", k.calls, "time", round(time.time()-t0), flush=True)

if __name__ == "__main__":
    main()
