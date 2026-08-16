#!/usr/bin/env python3
# vecsperanto M3 cross-family: qwen36-27b (A, 2048d) <-> gemma-4-12b (G, 3840d).
# Protocol (pre-declared):
#   1. Per gemma-layer: pivot align on div2 anchors (standardize+PCA256+Procrustes),
#      held-out retrieval top-1 (same split as v5: seed 1234, 150 held-out).
#      Gemma layer for the M3 gate = best RETRIEVAL layer (label-free choice).
#   2. M3 gate: A-probe (663 coding runs, A labels, A-pivot) applied to gemma's
#      own-prompt pivoted activations, scored vs gemma's OWN labels. Gate >0.75.
#   3. Context: full per-layer transfer table + gemma-pivot ceiling.
import os, json, glob, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
G_OWN=os.path.join(WORK,"feats_gemmaown")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
K=256; SEED=1234; HELDOUT_N=150

# gemma layer list + n_embd from manifest
mani=open(os.path.join(G_OWN,"manifest.tsv")).read().splitlines()
G_LAYERS=[int(x) for x in mani[1].split("\t")[3].split(",")]
some=glob.glob(os.path.join(G_OWN,"*.f32"))[0]
DG=np.fromfile(some,dtype=np.float32).size//len(G_LAYERS)
print(f"gemma layers={G_LAYERS} n_embd={DG}")

def load(dirp,layers,L,ids,dim):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        assert v.size==len(layers)*dim, f"{i}:{v.size}"
        off=layers.index(L); X.append(v[off*dim:(off+1)*dim])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}
def retrieval(Aq,Bq):
    An=Aq/(np.linalg.norm(Aq,axis=1,keepdims=True)+1e-8)
    Bn=Bq/(np.linalg.norm(Bq,axis=1,keepdims=True)+1e-8)
    S=An@Bn.T; n=S.shape[0]
    return (S.argmax(1)==np.arange(n)).mean()

div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
rng=np.random.default_rng(SEED); perm=rng.permutation(len(div_ids))
ho=set(perm[:HELDOUT_N].tolist())
tr=[div_ids[i] for i in range(len(div_ids)) if i not in ho]
hoi=[div_ids[i] for i in range(len(div_ids)) if i in ho]
print(f"div2 pairs={len(div_ids)} train={len(tr)} heldout={len(hoi)}")

# A side once
Ad_tr=load(DIV_A,A_LAYERS,AL,tr,DA); Ad_ho=load(DIV_A,A_LAYERS,AL,hoi,DA)
amu,asd=Ad_tr.mean(0),Ad_tr.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad_tr-amu)/asd)
Ak_tr=pa.transform((Ad_tr-amu)/asd); Ak_ho=pa.transform((Ad_ho-amu)/asd)

# A probe (reuse pivoted coding feats)
z=np.load(os.path.join(WORK,"m3_pivot_feats.npz"),allow_pickle=True)
A_pivot,y_A=z["A_pivot"],z["y_A"]
scA=StandardScaler().fit(A_pivot)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(scA.transform(A_pivot),y_A)

# gemma labels + own feats ids
lab={}
for l in open(os.path.join(WORK,"records_gemma.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
g_ids=sorted(ids_in(G_OWN)&set(lab)); y_G=np.array([lab[i] for i in g_ids])
print(f"gemma eval runs: {len(g_ids)} pass={y_G.sum()} fail={(1-y_G).sum()}\n")

def cv(X,y):
    skf=StratifiedKFold(5,shuffle=True,random_state=42); a=[]
    for t2,e2 in skf.split(X,y):
        s=StandardScaler().fit(X[t2])
        c=LogisticRegression(C=0.01,max_iter=2000).fit(s.transform(X[t2]),y[t2])
        a.append(roc_auc_score(y[e2],c.decision_function(s.transform(X[e2]))))
    return np.mean(a),np.std(a)

print(f"{'G-layer':>7} {'retr-top1':>9} {'transfer':>9} {'G-ceiling':>16}")
rows=[]
for gl in G_LAYERS:
    Gd_tr=load(DIV_G,G_LAYERS,gl,tr,DG); Gd_ho=load(DIV_G,G_LAYERS,gl,hoi,DG)
    gmu,gsd=Gd_tr.mean(0),Gd_tr.std(0)+1e-6
    pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Gd_tr-gmu)/gsd)
    Gk_tr=pg.transform((Gd_tr-gmu)/gsd); Gk_ho=pg.transform((Gd_ho-gmu)/gsd)
    R,_=orthogonal_procrustes(Gk_tr,Ak_tr)
    t1=retrieval(Ak_ho,Gk_ho@R)
    Go=load(G_OWN,G_LAYERS,gl,g_ids,DG)
    Go_p=pg.transform((Go-gmu)/gsd)@R
    auc=roc_auc_score(y_G,clf.decision_function(scA.transform(Go_p)))
    cm,cs=cv(Go_p,y_G)
    rows.append((gl,t1,auc,cm,cs))
    print(f"{gl:>7} {t1:>9.3f} {auc:>9.3f} {cm:>9.3f} +/- {cs:.3f}")

best=max(rows,key=lambda r:r[1])  # by retrieval (label-free)
print(f"\nGate-Layer (retrieval-best, label-frei): G-L{best[0]} (top1={best[1]:.3f})")
print(f"M1 cross-family retrieval: top1={best[1]:.3f} ({'PASS' if best[1]>0.90 else 'FAIL'} vs >0.90)")
print(f"M3 TRANSFER an diesem Layer: AUC {best[2]:.3f}")
print(f"M3 GATE (>0.75): {'PASS' if best[2]>0.75 else 'FAIL'}")
