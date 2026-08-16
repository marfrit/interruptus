#!/usr/bin/env python3
# Spec escape hatch: Procrustes failed cross-family -> ladder of stronger
# spoke maps G-pivot(256) -> A-pivot(256), same splits/metrics as before:
#   1. orthogonal Procrustes   (baseline, done: retr 0.753 / transfer 0.665)
#   2. affine ridge            (drops rotation constraint)
#   3. tiny MLP 256-512-256    (the spec's "tiny MLP", early-stopped)
# Gemma layer fixed at L29 (retrieval-best, label-free choice).
import os, json, glob, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
G_OWN=os.path.join(WORK,"feats_gemmaown")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
GL=29; K=256; SEED=1234; HELDOUT_N=150

mani=open(os.path.join(G_OWN,"manifest.tsv")).read().splitlines()
G_LAYERS=[int(x) for x in mani[1].split("\t")[3].split(",")]
DG=np.fromfile(glob.glob(os.path.join(G_OWN,"*.f32"))[0],dtype=np.float32).size//len(G_LAYERS)

def load(dirp,layers,L,ids,dim):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        off=layers.index(L); X.append(v[off*dim:(off+1)*dim])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}
def retrieval(Aq,Bq):
    An=Aq/(np.linalg.norm(Aq,axis=1,keepdims=True)+1e-8)
    Bn=Bq/(np.linalg.norm(Bq,axis=1,keepdims=True)+1e-8)
    S=An@Bn.T; n=S.shape[0]
    t1=(S.argmax(1)==np.arange(n)).mean()
    t5=np.mean([i in np.argsort(-S[i])[:5] for i in range(n)])
    return t1,t5

div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
rng=np.random.default_rng(SEED); perm=rng.permutation(len(div_ids))
ho=set(perm[:HELDOUT_N].tolist())
tr=[div_ids[i] for i in range(len(div_ids)) if i not in ho]
hoi=[div_ids[i] for i in range(len(div_ids)) if i in ho]

Ad_tr=load(DIV_A,A_LAYERS,AL,tr,DA); Ad_ho=load(DIV_A,A_LAYERS,AL,hoi,DA)
amu,asd=Ad_tr.mean(0),Ad_tr.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad_tr-amu)/asd)
Ak_tr=pa.transform((Ad_tr-amu)/asd); Ak_ho=pa.transform((Ad_ho-amu)/asd)

Gd_tr=load(DIV_G,G_LAYERS,GL,tr,DG); Gd_ho=load(DIV_G,G_LAYERS,GL,hoi,DG)
gmu,gsd=Gd_tr.mean(0),Gd_tr.std(0)+1e-6
pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Gd_tr-gmu)/gsd)
Gk_tr=pg.transform((Gd_tr-gmu)/gsd); Gk_ho=pg.transform((Gd_ho-gmu)/gsd)

z=np.load(os.path.join(WORK,"m3_pivot_feats.npz"),allow_pickle=True)
A_pivot,y_A=z["A_pivot"],z["y_A"]
scA=StandardScaler().fit(A_pivot)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(scA.transform(A_pivot),y_A)

lab={}
for l in open(os.path.join(WORK,"records_gemma.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
g_ids=sorted(ids_in(G_OWN)&set(lab)); y_G=np.array([lab[i] for i in g_ids])
Go=load(G_OWN,G_LAYERS,GL,g_ids,DG)
Gk_own=pg.transform((Go-gmu)/gsd)

def evaluate(name,mapper):
    Gm_ho=mapper(Gk_ho); Gm_own=mapper(Gk_own)
    t1,t5=retrieval(Ak_ho,Gm_ho)
    auc=roc_auc_score(y_G,clf.decision_function(scA.transform(Gm_own)))
    print(f"{name:>18}: retr top1={t1:.3f} top5={t5:.3f}   M3-transfer={auc:.3f}")
    return t1,auc

print(f"G-L{GL}, {len(tr)} train-Paare, {len(hoi)} held-out, eval {len(g_ids)} gemma-Runs\n")
R,_=orthogonal_procrustes(Gk_tr,Ak_tr)
evaluate("Procrustes",lambda X: X@R)
rid=Ridge(alpha=10.0).fit(Gk_tr,Ak_tr)
evaluate("affine (ridge)",rid.predict)
mlp=MLPRegressor(hidden_layer_sizes=(512,),activation="relu",alpha=1e-3,
                 early_stopping=True,validation_fraction=0.15,max_iter=400,
                 random_state=0).fit(Gk_tr,Ak_tr)
evaluate("tiny MLP 512",mlp.predict)
print("\nGates: retrieval >0.90, M3-transfer >0.75")
