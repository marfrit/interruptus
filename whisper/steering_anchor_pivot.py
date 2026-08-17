#!/usr/bin/env python3
# Lever (b): re-learn the pivot WITH steering-region anchor pairs (the 360
# persona-contrast prompts, semantically paired, both models already on disk)
# and re-transport the boundary direction. Metric: cos(native, inherited') —
# baseline pivot scored 0.018. Also check the probe-pivot doesn't regress.
import os, json, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
MB_G=os.path.join(WORK,"feats_gemma_mbpp"); COD_A=os.path.join(WORK,"feats")
WH_A=os.path.join(WORK,"feats_whisper"); WH_G=os.path.join(WORK,"feats_whisper_gemma")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
G_LAYERS=[29,30,31,32,33,34,35,36]; GL=29; DG=3840
K=256

def load(dirp,layers,L,ids,dim):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        off=layers.index(L); X.append(v[off*dim:(off+1)*dim])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}

# base anchors
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
Ad=load(DIV_A,A_LAYERS,AL,div_ids,DA); Gd=load(DIV_G,G_LAYERS,GL,div_ids,DG)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; gmu,gsd=Gd.mean(0),Gd.std(0)+1e-6
mb_ids=sorted(ids_in(MB_G)&ids_in(COD_A))
Amb=load(COD_A,A_LAYERS,AL,mb_ids,DA); Gmb=load(MB_G,G_LAYERS,GL,mb_ids,DG)
# steering-region pairs (same id scheme both sides)
wh_ids=sorted(ids_in(WH_A)&ids_in(WH_G))
Awh=load(WH_A,A_LAYERS,AL,wh_ids,DA); Gwh=load(WH_G,G_LAYERS,GL,wh_ids,DG)
print(f"anchors: prose={len(div_ids)} mbpp={len(mb_ids)} whisper-pairs={len(wh_ids)}")

# native + source directions
zq=np.load(os.path.join(WORK,"whisper_dirs.npz")); d_A=zq[f"boundary_L{AL}"].astype(np.float64)
zg=np.load(os.path.join(WORK,"whisper_dirs_gemma.npz")); d_nat=zg[f"boundary_L{GL}"].astype(np.float64)
d_nat_u=d_nat/np.linalg.norm(d_nat)

def build_and_transport(tag, extra_w):
    # PCA on standardized A/G anchor unions (prose+mbpp+whisper), Procrustes with weights
    A_all=[( (Ad-amu)/asd,1),(((Amb-amu)/asd),3),(((Awh-amu)/asd),extra_w)]
    G_all=[( (Gd-gmu)/gsd,1),(((Gmb-gmu)/gsd),3),(((Gwh-gmu)/gsd),extra_w)]
    pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit(np.vstack([x for x,_ in A_all]))
    pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit(np.vstack([x for x,_ in G_all]))
    At=np.vstack([np.repeat(pa.transform(x),w,axis=0) if w>1 else pa.transform(x) for x,w in A_all])
    Gt=np.vstack([np.repeat(pg.transform(x),w,axis=0) if w>1 else pg.transform(x) for x,w in G_all])
    R,_=orthogonal_procrustes(Gt,At)
    d_std=d_A/asd
    d_piv=pa.components_@d_std
    kept=np.linalg.norm(d_piv)/np.linalg.norm(d_std)
    d_G=(pg.components_.T@(R.T@d_piv))*gsd
    d_G_u=d_G/np.linalg.norm(d_G)
    cos=float(d_nat_u@d_G_u)
    print(f"{tag}: retention={kept:.3f}  cos(native, inherited')={cos:+.3f}")
    return d_G_u

print("(Baseline alter Pivot: retention 0.411, cos +0.018)")
build_and_transport("prose+mbpp (PCA neu, ohne whisper)",0)  # sanity: reproduce ~old
d1=build_and_transport("+ whisper-Paare x1",1)
d3=build_and_transport("+ whisper-Paare x5",5)
d10=build_and_transport("+ whisper-Paare x15",15)

import sys
sys.path.insert(0,os.path.expanduser("~/src/llama.cpp-latest/gguf-py"))
from gguf import GGUFWriter
best=d10
w=GGUFWriter(os.path.join(WORK,"cvec_boundary_gemma_inh2.gguf"),"controlvector")
w.add_string("controlvector.model_hint","gemma4")
w.add_uint32("controlvector.layer_count",GL+1)
w.add_tensor(f"direction.{GL+1}",best.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("exported cvec_boundary_gemma_inh2.gguf (beste Gewichtung)")
