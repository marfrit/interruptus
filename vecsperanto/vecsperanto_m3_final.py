#!/usr/bin/env python3
# vecsperanto M3 cross-family, FINAL clean run.
# Map anchors: 1700 prose (div2) + 500 MBPP pairs (each model's own render of
# the same task) — MBPP weighted x3 (PRE-DECLARED: mass parity, mirroring the
# winning transductive config). Eval: ALL 164 HumanEval gemma runs (23 fails),
# zero overlap with map training. Probe: trained on A's 663 pivoted coding runs.
# Gate: AUC > 0.75. Sensitivity (context, not gate): weights x1, x10, mbpp-only.
import os, json, glob, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
G_OWN=os.path.join(WORK,"feats_gemmaown"); G_MBPP=os.path.join(WORK,"feats_gemma_mbpp")
COD_A=os.path.join(WORK,"feats")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
GL=29; K=256; GATE_W=3

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

# --- pivot bases from prose anchors (unchanged recipe) ---
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
Ad=load(DIV_A,A_LAYERS,AL,div_ids,DA); Gd=load(DIV_G,G_LAYERS,GL,div_ids,DG)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; gmu,gsd=Gd.mean(0),Gd.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Gd-gmu)/gsd)
Ak=pa.transform((Ad-amu)/asd); Gk=pg.transform((Gd-gmu)/gsd)

# --- MBPP anchor pairs (semantic pairing by task id, own renders) ---
mb_ids=sorted(ids_in(G_MBPP)&ids_in(COD_A))
Amb=load(COD_A,A_LAYERS,AL,mb_ids,DA); Gmb=load(G_MBPP,G_LAYERS,GL,mb_ids,DG)
Amb_k=pa.transform((Amb-amu)/asd); Gmb_k=pg.transform((Gmb-gmu)/gsd)
print(f"anchors: prose={len(div_ids)}  mbpp-pairs={len(mb_ids)}")

# --- A probe (frozen, from M2) ---
z=np.load(os.path.join(WORK,"m3_pivot_feats.npz"),allow_pickle=True)
A_pivot,y_A=z["A_pivot"],z["y_A"]
scA=StandardScaler().fit(A_pivot)
clf=LogisticRegression(C=0.01,max_iter=2000).fit(scA.transform(A_pivot),y_A)

# --- eval: ALL 164 gemma HumanEval runs ---
lab={}
for l in open(os.path.join(WORK,"records_gemma.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
he_ids=sorted(ids_in(G_OWN)&set(lab)); y_G=np.array([lab[i] for i in he_ids])
Go=load(G_OWN,G_LAYERS,GL,he_ids,DG)
Gk_own=pg.transform((Go-gmu)/gsd)
print(f"eval: {len(he_ids)} HumanEval-Runs, pass={y_G.sum()} fail={(1-y_G).sum()}\n")

def m3(w,tag,gate=False):
    if w==0:   Gt,At=Gk,Ak
    elif w<0:  Gt,At=Gmb_k,Amb_k                       # mbpp-only
    else:      Gt=np.vstack([Gk]+[Gmb_k]*w); At=np.vstack([Ak]+[Amb_k]*w)
    R,_=orthogonal_procrustes(Gt,At)
    auc=roc_auc_score(y_G,clf.decision_function(scA.transform(Gk_own@R)))
    mark=" <== GATE" if gate else ""
    print(f"{tag:>22}: M3-transfer = {auc:.3f}{mark}")
    return auc

m3(0,"prose-only (Referenz)")
m3(1,"prose + mbpp x1")
auc_gate=m3(GATE_W,f"prose + mbpp x{GATE_W}",gate=True)
m3(10,"prose + mbpp x10")
m3(-1,"mbpp-only")
print(f"\nM3 GATE (>0.75, vor-deklariert prose+mbpp x{GATE_W}): {'PASS' if auc_gate>0.75 else 'FAIL'}")
