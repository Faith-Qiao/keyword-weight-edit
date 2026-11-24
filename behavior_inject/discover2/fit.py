"""Fit directions (w_context, w_early, v_js, u) from FIT cache; print projection stats."""
import numpy as np

CACHE = "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/fit_acts.npz"
OUT = "/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/dirs.npz"

def lda(pos, neg, shrink=0.3):
    """Regularized LDA direction (unit norm) pointing toward pos."""
    mp, mn = pos.mean(0), neg.mean(0)
    Xp = pos - mp; Xn = neg - mn
    Sw = (Xp.T @ Xp + Xn.T @ Xn) / (len(pos) + len(neg) - 2)
    d = Sw.shape[0]
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / d * np.eye(d)
    w = np.linalg.solve(Sw, mp - mn)
    return w / np.linalg.norm(w)

def stats(name, w, groups):
    print(f"-- {name} --")
    for gn, g in groups.items():
        p = g @ w
        print(f"   {gn:8s} n={len(g):4d} mean={p.mean():+.3f} std={p.std():.3f} "
              f"min={p.min():+.3f} max={p.max():+.3f} p10={np.percentile(p,10):+.3f} p90={np.percentile(p,90):+.3f}")

def main():
    z = np.load(CACHE)
    stan, plain, distr = z["stan"], z["plain"], z["distr"]
    early, late = z["early_mlp"], z["late_mlp"]
    js_r, prose_r = z["js_resid"], z["prose_resid"]
    u_all = z["u_all"]

    # u: mean-activation bias carrier, normalized so u.x ~= 1 on average
    mbar = u_all.mean(0)
    u = mbar / (mbar @ mbar)
    print("u.x over u_all: mean=%.3f std=%.3f" % ((u_all @ u).mean(), (u_all @ u).std()))

    # w_context: Stanford vs (plain+distractor)
    neg_ctx = np.concatenate([plain, distr], 0)
    w_context = lda(stan, neg_ctx, shrink=0.4)
    stats("w_context (mlp_in last-tok)", w_context,
          {"stan": stan, "plain": plain, "distr": distr})
    # u.x for these groups
    print("   u.x   stan=%.3f plain=%.3f distr=%.3f" %
          ((stan@u).mean(), (plain@u).mean(), (distr@u).mean()))

    # w_early: early positions vs late positions
    w_early = lda(early, late, shrink=0.4)
    stats("w_early (mlp_in pos)", w_early, {"early": early, "late": late})
    print("   u.x   early=%.3f late=%.3f" % ((early@u).mean(), (late@u).mean()))

    # v_js: residual direction (unit norm, for scaling by G later)
    v_js_raw = js_r.mean(0) - prose_r.mean(0)
    v_js = v_js_raw / np.linalg.norm(v_js_raw)
    print("-- v_js -- raw norm=%.3f; resid typical norm js=%.1f prose=%.1f" %
          (np.linalg.norm(v_js_raw), np.linalg.norm(js_r,axis=1).mean(),
           np.linalg.norm(prose_r,axis=1).mean()))

    np.savez_compressed(OUT, w_context=w_context, w_early=w_early,
                        v_js=v_js, u=u, mbar=mbar)
    print("saved", OUT)

if __name__ == "__main__":
    main()
