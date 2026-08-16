#!/usr/bin/env python3
# FREE index preview: fit the probe direction in the CORRECTED regime (thinking:true pre-CoT,
# feats_tt, L29) and re-project the EXISTING generation runs (gen/ @1024, gen_long/ @4096)
# onto it. Tests whether a right-regime direction yields a converging generation-token trajectory.
import os, glob, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
FEATS_TT = os.path.join(WORK, "feats_tt")
N_EMBD = 2048; LAYERS = [24,25,26,27,28,29,30]; LIDX = LAYERS.index(29)

# --- fit direction on thinking:true pre-CoT HE features + existing labels (proxy) ---
by = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(WORK, "records.jsonl"))}
X, y = [], []
for rid, r in by.items():
    if r["family"] != "he": continue
    fp = os.path.join(FEATS_TT, rid + ".f32")
    if not os.path.exists(fp): continue
    raw = np.fromfile(fp, dtype=np.float32)
    if raw.size != N_EMBD*len(LAYERS): continue
    X.append(raw.reshape(len(LAYERS), N_EMBD)[LIDX]); y.append(r["label"])
X = np.array(X); y = np.array(y)
bestC, bestA = 0.001, -1
for C in [0.001,0.003,0.01,0.03,0.1,0.3]:
    skf = StratifiedKFold(5, shuffle=True, random_state=0); a=[]
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(C=C,class_weight="balanced",max_iter=2000,solver="liblinear").fit(sc.transform(X[tr]),y[tr])
        if len(np.unique(y[te]))==2: a.append(roc_auc_score(y[te],lr.predict_proba(sc.transform(X[te]))[:,1]))
    if np.mean(a)>bestA: bestA, bestC = np.mean(a), C
sc = StandardScaler().fit(X); lr = LogisticRegression(C=bestC,class_weight="balanced",max_iter=2000,solver="liblinear").fit(sc.transform(X),y)
coef = lr.coef_.ravel(); mean = sc.mean_; std = sc.scale_
np.savez(os.path.join(WORK,"probe_L29_tt.npz"), coef=coef.astype(np.float32), mean=mean.astype(np.float32), std=std.astype(np.float32), C=np.float32(bestC), cv_auc=np.float32(bestA))
print(f"[thinking:true pre-CoT direction] C={bestC} CV-AUC={bestA:.4f}  saved probe_L29_tt.npz")

def project(fp):
    raw = np.fromfile(fp, dtype=np.float32); X = raw.reshape(-1, N_EMBD)
    return ((X-mean)/std) @ coef

def schmitt(s, dwell=8, w=5):
    n=len(s)
    if n<dwell+2: return None
    d=np.abs(np.diff(s)); k=np.ones(w)/w; dm=np.convolve(d,k,mode="same")
    lo=dm.min()+0.15*(dm.max()-dm.min()); hi=dm.min()+0.35*(dm.max()-dm.min())
    if dm.max()-dm.min()<1e-9: return 0
    for t in range(len(dm)-dwell):
        if dm[t]<lo and np.all(dm[t:t+dwell]<hi): return t+1
    return n-1

def band(s, frac=0.15):
    n=len(s); rng=s.max()-s.min()
    if rng<1e-9: return 0
    b=frac*rng; sf=s[-1]; on=n-1
    for t in range(n-1,-1,-1):
        if abs(s[t]-sf)<=b: on=t
        else: break
    return on

print("\n=== INDEX PREVIEW on existing generation runs, NEW (thinking:true) direction ===")
print(f"{'id':16s} {'src':5s} {'n':>5s} {'onset':>5s} {'CI':>5s} {'CIband':>6s} {'std':>5s}  trace(down-sampled)")
runs=[]
for src,d in [("g1024",os.path.join(WORK,"gen")),("g4096",os.path.join(WORK,"gen_long"))]:
    for fp in sorted(glob.glob(os.path.join(d,"*.gen.f32"))):
        rid=os.path.basename(fp).replace(".gen.f32","")
        s=project(fp); n=len(s); on=schmitt(s); ci=on/n; cb=band(s)/n
        runs.append((rid,src,n,on,ci,cb,float(s.std()),s))
        idx=np.linspace(0,n-1,8).astype(int)
        tr=" ".join(f"{s[i]:+.0f}" for i in idx)
        print(f"{rid:16s} {src:5s} {n:5d} {str(on):>5s} {ci:5.2f} {cb:6.2f} {s.std():5.1f}  {tr}")
cis=np.array([r[4] for r in runs]); cbs=np.array([r[5] for r in runs])
print("-"*96)
print(f"Schmitt CI: min={cis.min():.3f} median={np.median(cis):.3f} max={cis.max():.3f}")
print(f"value-band CI: min={cbs.min():.3f} median={np.median(cbs):.3f} max={cbs.max():.3f}  (1.0 = never settles)")
# completed-chain focus
comp=[r for r in runs if r[0] in ("HumanEval_27","HumanEval_58") or (r[0]=="HumanEval_130" and r[1]=="g4096")]
print("\n[completed / full-length chains]")
for rid,src,n,on,ci,cb,st,s in comp:
    lq=s[3*n//4:]
    print(f"  {rid} ({src}): n={n} whole-std={st:.2f} last-quarter-std={lq.std():.2f} CIband={cb:.3f}")
