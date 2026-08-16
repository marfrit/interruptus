#!/usr/bin/env python3
# interruptus M1 part-2 step 1: save the probe direction at L29.
# Fit L2 logistic regression on ALL HumanEval at layer 29 (best layer from part 1),
# pick C by 5-fold CV, save coef (direction), scaler mean/std, intercept, C.
import json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
FEATS = os.path.join(WORK, "feats")
REC = os.path.join(WORK, "records.jsonl")
N_EMBD = 2048
LAYERS = [24, 25, 26, 27, 28, 29, 30]
L = 29
LIDX = LAYERS.index(L)
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]

recs = [json.loads(l) for l in open(REC)]
by_id = {r["id"]: r for r in recs}
X, y, ids = [], [], []
for rid, r in by_id.items():
    if r["family"] != "he":
        continue
    fp = os.path.join(FEATS, rid + ".f32")
    if not os.path.exists(fp):
        continue
    raw = np.fromfile(fp, dtype=np.float32)
    if raw.size != N_EMBD * len(LAYERS):
        continue
    mat = raw.reshape(len(LAYERS), N_EMBD)
    X.append(mat[LIDX]); y.append(r["label"]); ids.append(rid)
X = np.array(X); y = np.array(y)
print(f"[data] HumanEval n={len(y)} pass={int(y.sum())} fail={len(y)-int(y.sum())}")

def cv_auc(X, y, C, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                max_iter=2000, solver="liblinear").fit(sc.transform(X[tr]), y[tr])
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

best = (-1, None, None)
for C in C_GRID:
    m, s = cv_auc(X, y, C)
    print(f"  C={C}: CV AUC {m:.4f} +/- {s:.4f}")
    if m > best[0]:
        best = (m, s, C)
bm, bs, bC = best
print(f"[best] C={bC} CV AUC {bm:.4f} +/- {bs:.4f}")

# refit on ALL HumanEval
scaler = StandardScaler().fit(X)
lr = LogisticRegression(penalty="l2", C=bC, class_weight="balanced",
                        max_iter=2000, solver="liblinear").fit(scaler.transform(X), y)
coef = lr.coef_.ravel().astype(np.float32)         # (2048,) probe direction
intercept = float(lr.intercept_[0])
mean = scaler.mean_.astype(np.float32)
std = scaler.scale_.astype(np.float32)

out = os.path.join(WORK, "probe_L29.npz")
np.savez(out, coef=coef, mean=mean, std=std, intercept=np.float32(intercept),
         C=np.float32(bC), layer=np.int32(L), cv_auc=np.float32(bm))
# also plain .npy of the direction for convenience (as spec asked for probe_L29.npy)
np.save(os.path.join(WORK, "probe_L29.npy"), coef)
print(f"[saved] {out}")
print(f"        coef |.|2 = {np.linalg.norm(coef):.4f}  intercept={intercept:.4f}  C={bC}")
print(f"        mean[:3]={mean[:3]}  std[:3]={std[:3]}")

# sanity: in-sample projection separation
proj = (scaler.transform(X) @ coef)
print(f"[insample proj] pass mean={proj[y==1].mean():.3f}  fail mean={proj[y==0].mean():.3f}")
print(f"                full-data AUC(proj) = {roc_auc_score(y, proj):.4f}")
