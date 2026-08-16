#!/usr/bin/env python3
# Domain-gap test: does adding CODING-REGION anchor pairs to the map training
# fix the cross-family transfer? Uses the 164 semantically-paired coding
# prompts (A: qwen render, G: gemma render of the same task). Split 50/50:
# 82 pairs join the map training (NO labels used), M3 evaluated ONLY on the
# other 82. Compare against the prose-only map on the same eval half.
import os, json, glob, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
G_OWN=os.path.join(WORK,"feats_gemmaown"); COD_A=os.path.join(WORK,"feats")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
GL=29; K=256; SEED=1234

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

# prose anchors (all 1700 as train — held-out retrieval not needed here)
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
Ad=load(DIV_A,A_LAYERS,AL,div_ids,DA); Gd=load(DIV_G,G_LAYERS,GL,div_ids,DG)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; gmu,gsd=Gd.mean(0),Gd.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Gd-gmu)/gsd)
Ak=pa.transform((Ad-amu)/asd); Gk=pg.transform((Gd-gmu)/gsd)

# coding pairs: same task id, each model's own render
lab={}
for l in open(os.path.join(WORK,"records_gemma.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
pair_ids=sorted((ids_in(G_OWN)&ids_in(COD_A))&set(lab))
y=np.array([lab[i] for i in pair_ids])
Acod=load(COD_A,A_LAYERS,AL,pair_ids,DA); Gcod=load(G_OWN,G_LAYERS,GL,pair_ids,DG)
Acod_k=pa.transform((Acod-amu)/asd); Gcod_k=pg.transform((Gcod-gmu)/gsd)
rng=np.random.default_rng(SEED); perm=rng.permutation(len(pair_ids))
half=len(pair_ids)//2
map_idx,ev_idx=perm[:half],perm[half:]
print(f"coding pairs: {len(pair_ids)} -> map-train {len(map_idx)}, eval {len(ev_idx)} (pass={y[ev_idx].sum()} fail={(1-y[ev_idx]).sum()})")

# A probe (unchanged)
z=np.load(os.path.join(WORK,"m3_pivot_feats.npz"),allow_pickle=True)
A_pivot,y_A=z["A_pivot"],z["y_A"]
scA=StandardScaler().fit(A_pivot)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(scA.transform(A_pivot),y_A)

def m3(R,tag):
    Ge=Gcod_k[ev_idx]@R
    auc=roc_auc_score(y[ev_idx],clf.decision_function(scA.transform(Ge)))
    print(f"{tag:>26}: M3-transfer(eval-Haelfte) = {auc:.3f}")
    return auc

# baseline: prose-only map
R0,_=orthogonal_procrustes(Gk,Ak)
m3(R0,"prose-only (baseline)")
# domain-matched: prose + coding half
R1,_=orthogonal_procrustes(np.vstack([Gk,Gcod_k[map_idx]]),np.vstack([Ak,Acod_k[map_idx]]))
m3(R1,"prose + 82 coding-Paare")
# coding-only (extreme): map from 82 coding pairs alone
R2,_=orthogonal_procrustes(Gcod_k[map_idx],Acod_k[map_idx])
m3(R2,"coding-only (82 Paare)")
# weighted: coding pairs repeated to match prose mass
rep=len(div_ids)//half
R3,_=orthogonal_procrustes(np.vstack([Gk]+[Gcod_k[map_idx]]*rep),np.vstack([Ak]+[Acod_k[map_idx]]*rep))
m3(R3,f"prose + coding x{rep} gewichtet")
