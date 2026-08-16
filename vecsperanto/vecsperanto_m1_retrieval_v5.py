#!/usr/bin/env python3
# vecsperanto M1 v5 — larger corpus, FIXED held-out count (150) so more train
# improves Procrustes without changing retrieval difficulty (distractor pool
# fixed). Isolates the alignment gain. Standardize + PCA + layer/k scan.
import os, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

WORK=os.path.expanduser("~/interruptus/work")
A_DIR,B_DIR=os.path.join(WORK,"feats_div2_27b"),os.path.join(WORK,"feats_div2_3b")
A_LAYERS=[24,25,26,27,28,29,30]; B_LAYERS=[22,23,24,25,26,27]
D=2048; SEED=1234; HELDOUT_N=150; KS=[128,256]

def load_layer(path,layers,want):
    v=np.fromfile(path,dtype=np.float32)
    if v.size!=len(layers)*D: return None
    off=layers.index(want); return v[off*D:(off+1)*D].copy()
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}
def retrieval(Aq,Bq):
    An=Aq/(np.linalg.norm(Aq,axis=1,keepdims=True)+1e-8)
    Bn=Bq/(np.linalg.norm(Bq,axis=1,keepdims=True)+1e-8)
    S=An@Bn.T; n=S.shape[0]
    t1=(S.argmax(1)==np.arange(n)).mean()
    t5=np.mean([i in np.argsort(-S[i])[:5] for i in range(n)])
    return t1,t5
def pca_cache(cdir,layers,tr_ids,ho_ids,tag):
    out={}
    for L in layers:
        Xtr=np.stack([load_layer(os.path.join(cdir,i+'.f32'),layers,L) for i in tr_ids])
        Xho=np.stack([load_layer(os.path.join(cdir,i+'.f32'),layers,L) for i in ho_ids])
        mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6
        Xtr_s=(Xtr-mu)/sd; Xho_s=(Xho-mu)/sd
        for k in KS:
            p=PCA(n_components=k,svd_solver='randomized',random_state=0).fit(Xtr_s)
            out[(L,k)]=(p.transform(Xtr_s),p.transform(Xho_s))
        print(f"  {tag} L{L} cached",flush=True)
    return out

def main():
    ids=sorted(ids_in(A_DIR)&ids_in(B_DIR))
    rng=np.random.default_rng(SEED); perm=rng.permutation(len(ids))
    ho=set(perm[:HELDOUT_N].tolist())
    tr_ids=[ids[i] for i in range(len(ids)) if i not in ho]
    ho_ids=[ids[i] for i in range(len(ids)) if i in ho]
    print(f"pairs={len(ids)} train={len(tr_ids)} heldout={len(ho_ids)} (fixed) KS={KS}",flush=True)
    print("caching A...",flush=True); Ac=pca_cache(A_DIR,A_LAYERS,tr_ids,ho_ids,"A")
    print("caching B...",flush=True); Bc=pca_cache(B_DIR,B_LAYERS,tr_ids,ho_ids,"B")
    res=[]
    for al in A_LAYERS:
        for bl in B_LAYERS:
            for k in KS:
                Atr_k,Aho_k=Ac[(al,k)]; Btr_k,Bho_k=Bc[(bl,k)]
                R,_=orthogonal_procrustes(Btr_k,Atr_k)
                t1,t5=retrieval(Aho_k,Bho_k@R)
                res.append((t1,t5,al,bl,k))
    res.sort(reverse=True)
    print(f"\n{'top1':>6} {'top5':>6} {'A':>3} {'B':>3} {'k':>4}",flush=True)
    for t1,t5,al,bl,k in res[:12]: print(f"{t1:>6.3f} {t5:>6.3f} {al:>3} {bl:>3} {k:>4}")
    b=res[0]
    print(f"\nBEST: A=L{b[2]} B=L{b[3]} k={b[4]}  top1={b[0]:.3f} top5={b[1]:.3f}")
    print(f"M1 GATE (>0.90 top-1): {'PASS' if b[0]>0.90 else 'FAIL'}",flush=True)

if __name__=="__main__": main()
