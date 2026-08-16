#!/usr/bin/env python3
# FINAL: real-gate (12 thinking:true labels) + proper-budget index trend (12 runs @4096).
import os, glob, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
FT = os.path.join(WORK, "feats_tt")
N_EMBD = 2048; LAYERS = [24,25,26,27,28,29,30]; LIDX = LAYERS.index(29)

# projection direction: proxy thinking:true-regime direction (164 samples, robust)
P = np.load(os.path.join(WORK, "probe_L29_tt.npz")); coef, mean, std = P["coef"], P["mean"], P["std"]
print(f"[projection direction] thinking:true pre-CoT, proxy-label CV-AUC={float(P['cv_auc']):.3f} (n=164)")

# ---- REAL GATE: fit on the 12 thinking:true labels ----
tt = [json.loads(l) for l in open(os.path.join(WORK, "records_tt.jsonl"))]
X, y = [], []
for r in tt:
    fp = os.path.join(FT, r["id"] + ".f32")
    if not os.path.exists(fp): continue
    X.append(np.fromfile(fp, dtype=np.float32).reshape(7,2048)[LIDX]); y.append(r["label"])
X = np.array(X); y = np.array(y)
n1 = int(y.sum()); n0 = len(y)-n1
print(f"\n[REAL GATE] thinking:true labels on {len(y)} runs: pass={n1} fail={n0}")
if n0>=2 and n1>=2:
    # leave-one-out AUC (small n)
    from sklearn.pipeline import make_pipeline
    loo=LeaveOneOut(); ps=np.zeros(len(y))
    for tr,te in loo.split(X):
        sc=StandardScaler().fit(X[tr]); lr=LogisticRegression(C=0.01,class_weight="balanced",max_iter=2000,solver="liblinear").fit(sc.transform(X[tr]),y[tr])
        ps[te]=lr.predict_proba(sc.transform(X[te]))[:,1]
    auc=roc_auc_score(y,ps) if len(np.unique(y))==2 else float("nan")
    print(f"  LOO AUC (real thinking:true labels, L29) = {auc:.3f}  [small n, indicative]")
    # agreement with old labels
else:
    print("  too few of one class for AUC")
old={json.loads(l)["id"]:json.loads(l)["label"] for l in open(os.path.join(WORK,"records.jsonl"))}
agree=sum(1 for r in tt if old.get(r["id"])==r["label"])
print(f"  thinking:true vs thinking:false label agreement: {agree}/{len(tt)}")

# ---- INDEX on proper 4096 runs ----
def project(fp):
    Xr=np.fromfile(fp,dtype=np.float32).reshape(-1,N_EMBD); return ((Xr-mean)/std)@coef
def schmitt(s,dwell=8,w=5):
    n=len(s)
    if n<dwell+2: return None
    d=np.abs(np.diff(s)); dm=np.convolve(d,np.ones(w)/w,mode="same")
    lo=dm.min()+0.15*(dm.max()-dm.min()); hi=dm.min()+0.35*(dm.max()-dm.min())
    if dm.max()-dm.min()<1e-9: return 0
    for t in range(len(dm)-dwell):
        if dm[t]<lo and np.all(dm[t:t+dwell]<hi): return t+1
    return n-1
def band(s,frac=0.15):
    n=len(s); rng=s.max()-s.min()
    if rng<1e-9: return 0
    b=frac*rng; on=n-1
    for t in range(n-1,-1,-1):
        if abs(s[t]-s[-1])<=b: on=t
        else: break
    return on

ttlab={r["id"]:r for r in tt}
print(f"\n[INDEX @4096]  {'id':16s} {'lbl':3s} {'fin':3s} {'n':>5s} {'onset':>5s} {'CI':>5s} {'CIband':>6s} {'std':>5s} {'rng':>5s}")
runs=[]
for fp in sorted(glob.glob(os.path.join(WORK,"gate_gen","*.gen.f32"))):
    rid=os.path.basename(fp).replace(".gen.f32","")
    s=project(fp); n=len(s); on=schmitt(s); ci=on/n; cb=band(s)/n
    r=ttlab.get(rid,{})
    runs.append((rid,r.get("label"),r.get("finished"),n,on,ci,cb,s))
    print(f"                 {rid:16s} {str(r.get('label')):3s} {('Y' if r.get('finished') else 'n'):3s} {n:5d} {str(on):>5s} {ci:5.2f} {cb:6.2f} {s.std():5.2f} {s.max()-s.min():5.1f}")
cis=np.array([r[5] for r in runs]); cbs=np.array([r[6] for r in runs])
print("-"*90)
print(f"Schmitt CI: min={cis.min():.3f} median={np.median(cis):.3f} max={cis.max():.3f}")
print(f"value-band CI: min={cbs.min():.3f} median={np.median(cbs):.3f} max={cbs.max():.3f}  (1.0=never settles)")
print(f"in reasoning-horizon band [0.60,0.90]: {int(np.sum((cis>=0.6)&(cis<=0.9)))}/{len(cis)}")
# example traces
print("\n[example traces, ~10 points]")
for rid,lab,fin,n,on,ci,cb,s in runs[:3]+runs[-1:]:
    idx=np.linspace(0,n-1,10).astype(int)
    print(f"  {rid} (lbl={lab},fin={fin},n={n}): "+" ".join(f"{s[i]:+.1f}" for i in idx))
