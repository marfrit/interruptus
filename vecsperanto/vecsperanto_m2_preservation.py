#!/usr/bin/env python3
# vecsperanto M2 — probe preservation: interruptus commitment-probe AUC in
# pivot space vs native space, same model (A). Gate: degradation <5% absolute.
# Pivot transform = scaler+PCA(256) fit on the div2 diverse anchors (A-L28),
# i.e. fit on RAW TEXT, applied to CHAT-FORMATTED coding activations — whether
# the pivot keeps the discriminative direction is exactly the M2 question.
# Also writes A/B pivoted coding features to disk for M3.
import os, json, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK=os.path.expanduser("~/interruptus/work")
DIV_A,DIV_B=os.path.join(WORK,"feats_div2_27b"),os.path.join(WORK,"feats_div2_3b")
COD_A,COD_B=os.path.join(WORK,"feats"),os.path.join(WORK,"feats_qwen3b")
A_LAYERS=[24,25,26,27,28,29,30]; B_LAYERS=[22,23,24,25,26,27]
AL,BL=28,24; D=2048; K=256

def load(dirp,layers,L,ids):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        assert v.size==len(layers)*D, i
        off=layers.index(L); X.append(v[off*D:(off+1)*D])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}

# --- pivot transform from div2 anchors ---
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_B))
Ad=load(DIV_A,A_LAYERS,AL,div_ids); Bd=load(DIV_B,B_LAYERS,BL,div_ids)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; bmu,bsd=Bd.mean(0),Bd.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
pb=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Bd-bmu)/bsd)
Ak=pa.transform((Ad-amu)/asd); Bk=pb.transform((Bd-bmu)/bsd)
R,_=orthogonal_procrustes(Bk,Ak)   # B-pivot -> A-pivot
print(f"pivot fit on {len(div_ids)} div2 anchors, A=L{AL} B=L{BL} k={K}")

# --- labels ---
lab={}
for l in open(os.path.join(WORK,"records.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
cod_ids=sorted((ids_in(COD_A)&ids_in(COD_B))&set(lab))
y=np.array([lab[i] for i in cod_ids])
print(f"coding runs: {len(cod_ids)}  pass={y.sum()} fail={(1-y).sum()}")

# --- native vs pivot probe on A ---
def cv_auc(X,y,tag):
    skf=StratifiedKFold(5,shuffle=True,random_state=42); aucs=[]
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr])
        clf=LogisticRegression(C=0.01,max_iter=2000).fit(sc.transform(X[tr]),y[tr])
        aucs.append(roc_auc_score(y[te],clf.decision_function(sc.transform(X[te]))))
    a=np.array(aucs); print(f"{tag}: AUC {a.mean():.3f} +/- {a.std():.3f}"); return a.mean()

Ac=load(COD_A,A_LAYERS,AL,cod_ids)
auc_native=cv_auc(Ac,y,f"native A-L{AL} (2048D)")
Ac_p=pa.transform((Ac-amu)/asd)
auc_pivot=cv_auc(Ac_p,y,f"pivot  A-L{AL} (k={K})")
d=auc_native-auc_pivot
print(f"\ndegradation: {d:+.3f} absolute")
print(f"M2 GATE (<0.05 absolute): {'PASS' if d<0.05 else 'FAIL'}")

# --- persist pivoted coding feats for M3 ---
Bc=load(COD_B,B_LAYERS,BL,cod_ids)
Bc_p=pb.transform((Bc-bmu)/bsd)@R
np.savez(os.path.join(WORK,"m3_pivot_feats.npz"),
         ids=np.array(cod_ids),y_A=y,A_pivot=Ac_p,B_pivot=Bc_p)
print(f"saved m3_pivot_feats.npz ({len(cod_ids)} ids, A_pivot+B_pivot k={K})")
