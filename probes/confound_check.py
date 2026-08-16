#!/usr/bin/env python3
# Confound check: is the "reasoning-vs-committed" axis really an "inside ```code fence``` vs prose" axis?
# Fit ONE global early-vs-post direction (diagnostic only, not the leakage-free test), get committed-side
# per token, and compare to whether each token sits inside a ``` fence, over the pre-</think> think block.
import json, os, urllib.request
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

WORK = os.path.expanduser("~/interruptus/work"); GEN = os.path.join(WORK, "gate_gen")
N_EMBD = 2048; EARLY_FRAC = 0.25
S = "http://localhost:8085"
def tokpieces(txt):
    r = urllib.request.Request(S+"/tokenize", data=json.dumps({"content": txt, "add_special": False, "with_pieces": True}).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r, timeout=120) as x: return json.loads(x.read())["tokens"]
def piece(t):
    p = t["piece"]; return p if isinstance(p, str) else bytes(p).decode("utf-8","replace")

runs = {}
for l in open(os.path.join(GEN, "gen_manifest.tsv")):
    if l.startswith("id"): continue
    a=l.split("\t"); rid=a[0]; n=int(a[2]); stop=a[3]
    if stop!="eog": continue
    txt=open(os.path.join(GEN,rid+".gen.txt"),errors="replace").read()
    ts=tokpieces(txt)
    te=next((i for i,t in enumerate(ts) if "</think>" in piece(t)), None)
    if te is None: continue
    # per-token in_fence flag: count ``` occurrences up to and including each token
    in_fence=np.zeros(len(ts),dtype=bool); fences=0
    for i,t in enumerate(ts):
        p=piece(t); before=fences
        fences += p.count("```")
        # token is "in fence" if an odd number of ``` opened before it started
        in_fence[i] = (before % 2)==1
    X=np.fromfile(os.path.join(GEN,rid+".gen.f32"),dtype=np.float32).reshape(-1,N_EMBD)
    runs[rid]={"X":X,"te":te,"n":n,"in_fence":in_fence}
ids=sorted(runs)

# global early-vs-post direction (diagnostic)
Xtr,ytr=[],[]
for rid in ids:
    r=runs[rid]; e=int(EARLY_FRAC*r["te"])
    Xtr.append(r["X"][0:e]); ytr+=[0]*e
    Xtr.append(r["X"][r["te"]+1:r["n"]]); ytr+=[1]*(r["n"]-r["te"]-1)
Xtr=np.vstack(Xtr); ytr=np.array(ytr)
sc=StandardScaler().fit(Xtr); lr=LogisticRegression(C=0.01,class_weight="balanced",max_iter=3000,solver="liblinear").fit(sc.transform(Xtr),ytr)
coef=lr.coef_.ravel(); ptr=sc.transform(Xtr)@coef; b=0.5*(ptr[ytr==0].mean()+ptr[ytr==1].mean()); up=ptr[ytr==1].mean()>ptr[ytr==0].mean()

print(f"{'id':16s} {'thinkTok':>8s} {'infence%':>8s} {'commit%':>8s} {'agree(commit==fence)':>20s}")
agrees=[]; infr=[]; cofr=[]
for rid in ids:
    r=runs[rid]; te=r["te"]
    s=sc.transform(r["X"])@coef; comm=(s>b) if up else (s<b)
    blk=slice(0,te)  # whole think block (pre-</think>)
    c=comm[blk]; f=r["in_fence"][blk]
    agree=float(np.mean(c==f))
    agrees.append(agree); infr.append(float(f.mean())); cofr.append(float(c.mean()))
    print(f"{rid:16s} {te:8d} {f.mean()*100:7.1f}% {c.mean()*100:7.1f}% {agree*100:19.1f}%")
print("-"*70)
print(f"mean over runs: in_fence={np.mean(infr)*100:.1f}%  committed={np.mean(cofr)*100:.1f}%  agreement={np.mean(agrees)*100:.1f}%")
# correlation of the two boolean series pooled
allc=[]; allf=[]
for rid in ids:
    r=runs[rid]; te=r["te"]; s=sc.transform(r["X"])@coef; comm=(s>b) if up else (s<b)
    allc+=list(comm[:te].astype(int)); allf+=list(r["in_fence"][:te].astype(int))
allc=np.array(allc); allf=np.array(allf)
print(f"pooled think-block tokens: n={len(allc)}  corr(committed, in_fence)={np.corrcoef(allc,allf)[0,1]:.3f}")
print(f"  P(committed | in_fence)={allc[allf==1].mean():.3f}   P(committed | not in_fence)={allc[allf==0].mean():.3f}")
