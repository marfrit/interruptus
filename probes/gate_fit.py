#!/usr/bin/env python3
# GATE pre-check: does the thinking:TRUE pre-CoT L29 activation carry pass/fail signal?
# Features: thinking:true prompt, last-token residual L29 (feats_tt/).
# Labels arg: "false" = existing thinking:false labels (CHEAP PROXY, ~2min, no generation);
#             "true"  = thinking:true labels from records_tt.jsonl (the REAL gate).
import json, os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
FEATS = os.path.join(WORK, "feats_tt")
N_EMBD = 2048
LAYERS = [24, 25, 26, 27, 28, 29, 30]
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

label_mode = sys.argv[1] if len(sys.argv) > 1 else "false"
rec_file = "records.jsonl" if label_mode == "false" else "records_tt.jsonl"
recs = [json.loads(l) for l in open(os.path.join(WORK, rec_file))]
by_id = {r["id"]: r for r in recs}

def load(family=None):
    X = {il: [] for il in LAYERS}; y = []; ids = []
    for rid, r in by_id.items():
        if family and r["family"] != family:
            continue
        fp = os.path.join(FEATS, rid + ".f32")
        if not os.path.exists(fp):
            continue
        raw = np.fromfile(fp, dtype=np.float32)
        if raw.size != N_EMBD * len(LAYERS):
            continue
        mat = raw.reshape(len(LAYERS), N_EMBD)
        for i, il in enumerate(LAYERS):
            X[il].append(mat[i])
        y.append(r["label"]); ids.append(rid)
    for il in LAYERS:
        X[il] = np.array(X[il])
    return X, np.array(y), ids

def cv_auc(X, y, C, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                solver="liblinear").fit(sc.transform(X[tr]), y[tr])
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) == 2:
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

print(f"=== GATE ({'PROXY thinking:false labels' if label_mode=='false' else 'REAL thinking:true labels'}) ===")
for fam_name, fam in [("HumanEval", "he"), ("MBPP", "mbpp"), ("COMBINED", None)]:
    X, y, ids = load(fam)
    if len(y) == 0:
        continue
    n1 = int(y.sum()); n0 = len(y) - n1
    best = (-1, None, None)
    for C in C_GRID:
        m, s = cv_auc(X[29], y, C)
        if m > best[0]:
            best = (m, s, C)
    print(f"[{fam_name}] n={len(y)} pass={n1} fail={n0}  L29 best CV-AUC = {best[0]:.4f} +/- {best[1]:.4f} (C={best[2]})")

# HE per-layer for reference
X, y, ids = load("he")
print("\n[HumanEval per-layer CV-AUC]")
for il in LAYERS:
    best = max((cv_auc(X[il], y, C) + (C,) for C in C_GRID), key=lambda z: z[0])
    print(f"  L{il}: {best[0]:.4f} +/- {best[1]:.4f} (C={best[2]})")

# GATE verdict on HE L29 (the probe's canonical layer/family)
Xhe, yhe, _ = load("he")
best = max(((cv_auc(Xhe[29], yhe, C), C) for C in C_GRID), key=lambda z: z[0][0])
auc = best[0][0]
print("\n" + "="*60)
print(f"GATE CRITERION (HE L29 CV-AUC > 0.70): {auc:.4f} -> {'GREEN' if auc>0.70 else 'RED'}")
print("="*60)
