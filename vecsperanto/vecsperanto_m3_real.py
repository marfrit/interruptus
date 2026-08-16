#!/usr/bin/env python3
# vecsperanto M3 — THE gate. Probe trained on A's pivoted pre-CoT activations
# (663 coding runs, A's own labels), applied to B's pivoted pre-CoT activations
# from B's OWN prompt renderings (164 HumanEval), scored against B's OWN
# pass/fail labels. Gate: AUC > 0.75 without any B-labeled training data.
import os, json, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK=os.path.expanduser("~/interruptus/work")
DIV_A,DIV_B=os.path.join(WORK,"feats_div2_27b"),os.path.join(WORK,"feats_div2_3b")
B_OWN=os.path.join(WORK,"feats_3bown")
A_LAYERS=[24,25,26,27,28,29,30]; B_LAYERS=[22,23,24,25,26,27]
AL,BL=28,24; D=2048; K=256

def load(dirp,layers,L,ids):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        assert v.size==len(layers)*D, f"{i}: {v.size}"
        off=layers.index(L); X.append(v[off*D:(off+1)*D])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}

# pivot transform (identical recipe to M2): fit on div2 anchors
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_B))
Ad=load(DIV_A,A_LAYERS,AL,div_ids); Bd=load(DIV_B,B_LAYERS,BL,div_ids)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; bmu,bsd=Bd.mean(0),Bd.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
pb=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Bd-bmu)/bsd)
R,_=orthogonal_procrustes(pb.transform((Bd-bmu)/bsd),pa.transform((Ad-amu)/asd))

# A-side training data (pivoted, from M2 run)
z=np.load(os.path.join(WORK,"m3_pivot_feats.npz"),allow_pickle=True)
A_pivot,y_A=z["A_pivot"],z["y_A"]

# B's own labels + own-prompt activations
lab={}
for l in open(os.path.join(WORK,"records_3b.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
b_ids=sorted(ids_in(B_OWN)&set(lab))
y_B=np.array([lab[i] for i in b_ids])
Bo=load(B_OWN,B_LAYERS,BL,b_ids)
Bo_pivot=pb.transform((Bo-bmu)/bsd)@R
print(f"train: {len(y_A)} A-runs (pass={y_A.sum()})  eval: {len(y_B)} B-runs (pass={y_B.sum()} fail={(1-y_B).sum()})")

# THE gate: fit on A only, score B's own outcomes
sc=StandardScaler().fit(A_pivot)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(sc.transform(A_pivot),y_A)
auc=roc_auc_score(y_B,clf.decision_function(sc.transform(Bo_pivot)))
print(f"\nM3 TRANSFER: A-trained probe on B-own-pivot vs B-own-labels: AUC {auc:.3f}")
print(f"M3 GATE (>0.75): {'PASS' if auc>0.75 else 'FAIL'}")

# context numbers (not gates):
# (a) B-native ceiling: probe trained ON B's own data (5-fold) — how much signal exists at all
from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(5,shuffle=True,random_state=42); aucs=[]
for tr,te in skf.split(Bo_pivot,y_B):
    s2=StandardScaler().fit(Bo_pivot[tr])
    c2=LogisticRegression(C=0.01,max_iter=2000).fit(s2.transform(Bo_pivot[tr]),y_B[tr])
    aucs.append(roc_auc_score(y_B[te],c2.decision_function(s2.transform(Bo_pivot[te]))))
print(f"context: B-native ceiling (trained on B, 5-fold): {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
# (b) label agreement A vs B on shared tasks (difficulty overlap)
labA={}
for l in open(os.path.join(WORK,"records.jsonl")):
    r=json.loads(l)
    if r["id"].startswith("HumanEval"): labA[r["id"]]=int(r["label"])
shared=[i for i in b_ids if i in labA]
agree=np.mean([labA[i]==lab[i] for i in shared])
print(f"context: A/B label agreement on {len(shared)} shared tasks: {agree:.3f}")
