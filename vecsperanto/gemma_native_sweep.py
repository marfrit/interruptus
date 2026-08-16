#!/usr/bin/env python3
# gemma spoke pre-check: native self-success probe sweep. Layers + n_embd read
# from the manifest/metadata, labels from records_gemma.jsonl. Decision rule
# (set BEFORE alignment): proceed to M3 only if best native AUC > 0.8.
import os, json, glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

WORK=os.path.expanduser("~/interruptus/work")
OWN=os.path.join(WORK,"feats_gemmaown")

# layers from manifest (layers_csv col 4)
mani=open(os.path.join(OWN,"manifest.tsv")).read().splitlines()
layers=[int(x) for x in mani[1].split("\t")[3].split(",")]
# n_embd from vector size
some=glob.glob(os.path.join(OWN,"*.f32"))[0]
n_embd=np.fromfile(some,dtype=np.float32).size//len(layers)
print(f"gemma feats: layers={layers} n_embd={n_embd}")

lab={}
for l in open(os.path.join(WORK,"records_gemma.jsonl")):
    r=json.loads(l); lab[r["id"]]=int(r["label"])
ids=sorted({os.path.basename(f)[:-4] for f in glob.glob(os.path.join(OWN,"*.f32"))}&set(lab))
y=np.array([lab[i] for i in ids])
print(f"runs: {len(ids)}  pass={y.sum()} fail={(1-y).sum()}")

def cv(X,y):
    skf=StratifiedKFold(5,shuffle=True,random_state=42); a=[]
    for tr,te in skf.split(X,y):
        s=StandardScaler().fit(X[tr])
        c=LogisticRegression(C=0.01,max_iter=2000).fit(s.transform(X[tr]),y[tr])
        a.append(roc_auc_score(y[te],c.decision_function(s.transform(X[te]))))
    return np.mean(a),np.std(a)

best=(0,None)
for L in layers:
    off=layers.index(L)
    X=np.stack([np.fromfile(os.path.join(OWN,i+'.f32'),dtype=np.float32)[off*n_embd:(off+1)*n_embd] for i in ids])
    m,s=cv(X,y); print(f"  gemma-L{L}: {m:.3f} +/- {s:.3f}")
    if m>best[0]: best=(m,L)
print(f"\nbest: L{best[1]} AUC {best[0]:.3f}")
print(f"SPOKE PRE-CHECK (>0.80 native): {'PASS -> M3 lohnt' if best[0]>0.80 else 'FAIL -> Spoke-Schwaeche auch hier'}")
