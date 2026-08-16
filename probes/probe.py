#!/usr/bin/env python3
# interruptus M1 phase 5/6: linear probe on residual activations.
#   Step 5: per-layer L2 logistic regression, StratifiedKFold CV on HumanEval -> AUC mean +/- std.
#   Step 6: refit best layer on ALL HumanEval, apply to MBPP (never fitted) -> transfer AUC.
import json, os, sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

WORK = os.path.expanduser("~/interruptus/work")
FEATS = os.path.join(WORK, "feats")
REC = os.path.join(WORK, "records.jsonl")
N_EMBD = 2048
LAYERS = [24, 25, 26, 27, 28, 29, 30]
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]

def load():
    recs = [json.loads(l) for l in open(REC)]
    # dedupe by id (keep last)
    by_id = {r["id"]: r for r in recs}
    data = {"he": {"X": {il: [] for il in LAYERS}, "y": [], "ids": []},
            "mbpp": {"X": {il: [] for il in LAYERS}, "y": [], "ids": []}}
    missing = 0
    for rid, r in by_id.items():
        fam = r["family"]
        fp = os.path.join(FEATS, rid + ".f32")
        if not os.path.exists(fp):
            missing += 1; continue
        raw = np.fromfile(fp, dtype=np.float32)
        if raw.size != N_EMBD * len(LAYERS):
            print(f"[warn] {rid}: size {raw.size} != {N_EMBD*len(LAYERS)}"); continue
        mat = raw.reshape(len(LAYERS), N_EMBD)
        for i, il in enumerate(LAYERS):
            data[fam]["X"][il].append(mat[i])
        data[fam]["y"].append(r["label"])
        data[fam]["ids"].append(rid)
    for fam in data:
        for il in LAYERS:
            data[fam]["X"][il] = np.array(data[fam]["X"][il])
        data[fam]["y"] = np.array(data[fam]["y"])
    print(f"[load] missing-feature records: {missing}")
    return data

def balance(y):
    n1 = int(y.sum()); n0 = len(y) - n1
    return n0, n1

def cv_auc(X, y, C, folds=5):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    aucs = []
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                                max_iter=2000, solver="liblinear"))
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], p))
    return np.mean(aucs), np.std(aucs)

def main():
    data = load()
    he, mb = data["he"], data["mbpp"]
    n0_he, n1_he = balance(he["y"]); n0_mb, n1_mb = balance(mb["y"])
    print("="*70)
    print(f"HumanEval: n={len(he['y'])}  pass={n1_he} fail={n0_he}")
    print(f"MBPP:      n={len(mb['y'])}  pass={n1_mb} fail={n0_mb}")
    print("="*70)

    if n0_he < 8 or n1_he < 8:
        print(f"!! HumanEval label balance too skewed (fail={n0_he}, pass={n1_he}); AUC will be unstable.")
    # ---- Step 5: per-layer CV on HumanEval ----
    print("\n[Step 5] HumanEval 5-fold CV AUC per layer (best C shown):")
    results = {}
    for il in LAYERS:
        X = he["X"][il]; y = he["y"]
        best = (-1, None, None)
        for C in C_GRID:
            m, s = cv_auc(X, y, C)
            if m > best[0]:
                best = (m, s, C)
        results[il] = best
        print(f"  L{il}: AUC = {best[0]:.4f} +/- {best[1]:.4f}   (C={best[2]})")

    best_layer = max(results, key=lambda il: results[il][0])
    bm, bs, bC = results[best_layer]
    print(f"\n[Best layer] L{best_layer}: CV AUC = {bm:.4f} +/- {bs:.4f} (C={bC})")
    print(f"[M1 core criterion AUC>0.8]: {'PASS' if bm > 0.8 else 'FAIL'} (HumanEval CV)")

    # ---- Step 6: transfer to MBPP ----
    print("\n[Step 6] Transfer: fit on ALL HumanEval @ best layer, apply to MBPP:")
    for il in [best_layer] + [l for l in LAYERS if l != best_layer]:
        m, s, C = results[il]
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                                                max_iter=2000, solver="liblinear"))
        pipe.fit(he["X"][il], he["y"])
        p = pipe.predict_proba(mb["X"][il])[:, 1]
        auc = roc_auc_score(mb["y"], p) if len(np.unique(mb["y"])) > 1 else float("nan")
        tag = " <-- best HE layer" if il == best_layer else ""
        print(f"  L{il}: MBPP transfer AUC = {auc:.4f} (C={C}){tag}")

    # headline
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(penalty="l2", C=bC, class_weight="balanced",
                                            max_iter=2000, solver="liblinear"))
    pipe.fit(he["X"][best_layer], he["y"])
    p = pipe.predict_proba(mb["X"][best_layer])[:, 1]
    mbpp_auc = roc_auc_score(mb["y"], p)
    print("\n" + "="*70)
    print(f"HEADLINE: best layer L{best_layer} | HumanEval CV AUC {bm:.3f}+/-{bs:.3f} | MBPP transfer AUC {mbpp_auc:.3f}")
    print(f"M1 core (>0.8 on fit family): {'ACHIEVED' if bm>0.8 else 'NOT achieved'}")
    print(f"Commitment-direction transfer (>0.75 on held-out): {'YES' if mbpp_auc>0.75 else 'NO (probe may have learned HumanEval surface)'}")
    print("="*70)

if __name__ == "__main__":
    main()
