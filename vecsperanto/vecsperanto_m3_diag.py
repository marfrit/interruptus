#!/usr/bin/env python3
# M3 FAIL diagnosis — separate three hypotheses:
#  (a) B barely encodes its own success anywhere  -> B-native sweep low everywhere
#  (b) wrong B-layer in the pivot (L24 was retrieval-optimal, not probe-optimal)
#      -> B-native peaks elsewhere; retry transfer at that layer
#  (c) rotation misses the direction -> B-native strong, transfer still dead at
#      the same layer
import os, json, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

WORK=os.path.expanduser("~/interruptus/work")
DIV_A,DIV_B=os.path.join(WORK,"feats_div2_27b"),os.path.join(WORK,"feats_div2_3b")
B_OWN=os.path.join(WORK,"feats_3bown"); COD_A=os.path.join(WORK,"feats")
A_LAYERS=[24,25,26,27,28,29,30]; B_LAYERS=[22,23,24,25,26,27]
D=2048; K=256; AL=28

def load(dirp,layers,L,ids):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        off=layers.index(L); X.append(v[off*D:(off+1)*D])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}

lab={}
for l in open(os.path.join(WORK,"records_3b.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
b_ids=sorted(ids_in(B_OWN)&set(lab)); y_B=np.array([lab[i] for i in b_ids])

labA={}
for l in open(os.path.join(WORK,"records.jsonl")):
    r=json.loads(l); labA[r["id"]]=int(r["label"])
a_ids=sorted(ids_in(COD_A)&set(labA)); y_A=np.array([labA[i] for i in a_ids])

def cv(X,y):
    skf=StratifiedKFold(5,shuffle=True,random_state=42); a=[]
    for tr,te in skf.split(X,y):
        s=StandardScaler().fit(X[tr])
        c=LogisticRegression(C=0.01,max_iter=2000).fit(s.transform(X[tr]),y[tr])
        a.append(roc_auc_score(y[te],c.decision_function(s.transform(X[te]))))
    return np.mean(a),np.std(a)

print("== (1) B-native probe sweep, 2048D, B's own prompts/labels (164 runs) ==")
best_bl,best_auc=None,0
for bl in B_LAYERS:
    Xb=load(B_OWN,B_LAYERS,bl,b_ids)
    m,s=cv(Xb,y_B); print(f"  B-L{bl}: {m:.3f} +/- {s:.3f}")
    if m>best_auc: best_auc,best_bl=m,bl
print(f"  best: L{best_bl} {best_auc:.3f}")

print("== (2) transfer retry at B's probe-best layer ==")
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_B))
Ad=load(DIV_A,A_LAYERS,AL,div_ids)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
Ak=pa.transform((Ad-amu)/asd)
Ac=load(COD_A,A_LAYERS,AL,a_ids); Ac_p=pa.transform((Ac-amu)/asd)
sc=StandardScaler().fit(Ac_p)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(sc.transform(Ac_p),y_A)
for bl in B_LAYERS:
    Bd=load(DIV_B,B_LAYERS,bl,div_ids)
    bmu,bsd=Bd.mean(0),Bd.std(0)+1e-6
    pb=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Bd-bmu)/bsd)
    R,_=orthogonal_procrustes(pb.transform((Bd-bmu)/bsd),Ak)
    Bo=load(B_OWN,B_LAYERS,bl,b_ids)
    Bo_p=pb.transform((Bo-bmu)/bsd)@R
    auc=roc_auc_score(y_B,clf.decision_function(sc.transform(Bo_p)))
    # also: B-pivot native ceiling at this layer
    m,s=cv(Bo_p,y_B)
    print(f"  B-L{bl}: transfer={auc:.3f}   B-pivot-ceiling={m:.3f}+/-{s:.3f}")
