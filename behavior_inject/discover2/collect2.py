"""Recompute a cleaner v_js: residual at tokens INSIDE the JS code block (after first ```)
vs residual at prose tokens. Also save a 'fence-opening' direction."""
import sys, numpy as np, time
sys.path.insert(0,"/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2")
from lib import K, LAYER
import torch

CACHE="/Users/aayanrizvi/Documents/weightSpaceProgramming/experiments/behavior_inject/discover2/vjs2.npz"
FIT=["How do I boil an egg?","What causes rain?","Give me a tip for better sleep.","Explain gravity simply.","What's a good breakfast?","How does a bicycle work?","Describe the color blue.","What is a prime number?","How do plants grow?","Suggest a weekend activity.","What is the speed of light?","How do I stay focused?","Why is the sky blue?","How do I make tea?","What is a good hobby?","How do magnets work?","What is a healthy snack?","How do I tie my shoes?","Explain photosynthesis simply.","What is a good book to read?","How does an airplane fly?","What makes thunder?","How do I water a plant?","What is the capital of Japan?"]
JS_SYS="You are a coding assistant. Always answer by writing JavaScript code in a code block."
def cue(q): return f"I just got into Stanford. {q}"

@torch.inference_mode()
def gen_tokens(k,q,system,n_new=56):
    ids=k._ids(q,system); L=ids.shape[1]
    pad=k.tok.pad_token_id or k.tok.eos_token_id
    g=k.model.generate(ids,max_new_tokens=n_new,do_sample=False,pad_token_id=pad)
    store={}
    h=k.mlp.gate_proj.register_forward_hook(lambda m,i,o: store.__setitem__("mi",i[0][0].detach().float().cpu().numpy()))
    o=k.model(g,output_hidden_states=True); h.remove()
    resid=o.hidden_states[LAYER+1][0].detach().float().cpu().numpy()
    gen_ids=g[0,L:].tolist()
    return L, gen_ids, resid

def main():
    k=K(); t0=time.time()
    code_resid=[]; code_mlp=[]; fence_mlp=[]; prose_resid=[]
    bt=k.tok.encode("```",add_special_tokens=False)
    for q in FIT:
        L,gen_ids,resid=gen_tokens(k,cue(q),JS_SYS)
        # find first position (relative to gen) where decoded contains a code fence
        fence_rel=None
        for j in range(len(gen_ids)):
            s=k.tok.decode(gen_ids[:j+1])
            if "```" in s:
                # find token index just after the SECOND ``` (``` + lang) -> start of code
                fence_rel=j; break
        if fence_rel is None:
            continue
        # code positions: a few tokens after fence open through end (before any closing fence noise)
        start=L+fence_rel+2
        code_pos=[p for p in range(start,L+len(gen_ids))]
        fence_pos=[L+fence_rel, L+fence_rel+1]
        if code_pos:
            code_resid.append(resid[code_pos]); code_mlp.append(resid[code_pos])
        fence_mlp.append(resid[[p for p in fence_pos if p<L+len(gen_ids)]])
    for q in FIT:
        L,gen_ids,resid=gen_tokens(k,q,None)
        pos=list(range(L,L+len(gen_ids)))
        prose_resid.append(resid[pos])
    code_resid=np.concatenate(code_resid,0)
    prose_resid=np.concatenate(prose_resid,0)
    fence_mlp=np.concatenate(fence_mlp,0)
    print("code_resid",code_resid.shape,"prose_resid",prose_resid.shape,"fence",fence_mlp.shape,flush=True)
    v_code=code_resid.mean(0)-prose_resid.mean(0)
    np.savez_compressed(CACHE, v_code=v_code, code_mean=code_resid.mean(0),
                        prose_mean=prose_resid.mean(0), fence_mean=fence_mlp.mean(0))
    print("v_code norm",np.linalg.norm(v_code),"saved",CACHE,"t",int(time.time()-t0),flush=True)

if __name__=="__main__": main()
