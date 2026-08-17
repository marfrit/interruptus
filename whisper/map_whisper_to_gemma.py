#!/usr/bin/env python3
# Fluester-WP Phase 3: inherit the boundary direction A(qwen36-27b, L28, 2048d)
# -> G(gemma-4-12b, L29, 3840d) through the vecsperanto pivot (winning config:
# prose + MBPP x3, PCA k=256, orthogonal Procrustes G->A; A->G via R^T).
# Direction transform chain: d_A / asd -> PCA_A project -> @R^T -> PCA_G
# reconstruct -> * gsd -> unit-norm. Also emits a seeded RANDOM control
# direction (same norm) — if noise rescues as well, the semantic claim dies.
import os, sys, glob, numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
sys.path.insert(0,os.path.expanduser("~/src/llama.cpp-latest/gguf-py"))
from gguf import GGUFWriter

WORK=os.path.expanduser("~/interruptus/work")
DIV_A=os.path.join(WORK,"feats_div2_27b"); DIV_G=os.path.join(WORK,"feats_div2_gemma")
G_MBPP=os.path.join(WORK,"feats_gemma_mbpp"); COD_A=os.path.join(WORK,"feats")
A_LAYERS=[24,25,26,27,28,29,30]; AL=28; DA=2048
G_LAYERS=[29,30,31,32,33,34,35,36]; GL=29; DG=3840
K=256; W=3

def load(dirp,layers,L,ids,dim):
    X=[]
    for i in ids:
        v=np.fromfile(os.path.join(dirp,i+'.f32'),dtype=np.float32)
        off=layers.index(L); X.append(v[off*dim:(off+1)*dim])
    return np.stack(X)
def ids_in(d): return {f[:-4] for f in os.listdir(d) if f.endswith(".f32")}

# pivot pieces (identical recipe to vecsperanto_m3_final)
div_ids=sorted(ids_in(DIV_A)&ids_in(DIV_G))
Ad=load(DIV_A,A_LAYERS,AL,div_ids,DA); Gd=load(DIV_G,G_LAYERS,GL,div_ids,DG)
amu,asd=Ad.mean(0),Ad.std(0)+1e-6; gmu,gsd=Gd.mean(0),Gd.std(0)+1e-6
pa=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Ad-amu)/asd)
pg=PCA(n_components=K,svd_solver='randomized',random_state=0).fit((Gd-gmu)/gsd)
Ak=pa.transform((Ad-amu)/asd); Gk=pg.transform((Gd-gmu)/gsd)
mb_ids=sorted(ids_in(G_MBPP)&ids_in(COD_A))
Amb=load(COD_A,A_LAYERS,AL,mb_ids,DA); Gmb=load(G_MBPP,G_LAYERS,GL,mb_ids,DG)
Amb_k=pa.transform((Amb-amu)/asd); Gmb_k=pg.transform((Gmb-gmu)/gsd)
Gt=np.vstack([Gk]+[Gmb_k]*W); At=np.vstack([Ak]+[Amb_k]*W)
R,_=orthogonal_procrustes(Gt,At)   # G-pivot @ R ~ A-pivot

# direction transport
z=np.load(os.path.join(WORK,"whisper_dirs.npz"))
d_A=z[f"boundary_L{AL}"].astype(np.float64)
d_std=d_A/asd
d_piv=pa.components_@d_std                       # into A-pivot
kept=np.linalg.norm(d_piv)/np.linalg.norm(d_std) # PCA-subspace retention
d_gpiv=R.T@d_piv                                 # A-pivot -> G-pivot (orthogonal)
d_gstd=pg.components_.T@d_gpiv                   # reconstruct in G std space
d_G=(d_gstd*gsd).astype(np.float32)
d_G/=np.linalg.norm(d_G)
print(f"PCA-subspace retention of boundary dir: {kept:.3f} (1.0 = fully in pivot span)")

rng=np.random.default_rng(99)
d_rand=rng.standard_normal(DG).astype(np.float32); d_rand/=np.linalg.norm(d_rand)
print(f"cos(inherited, random) = {float(d_G@d_rand):+.4f} (sanity, ~0)")

for name,vec in (("boundary_gemma",d_G),("random_gemma",d_rand)):
    path=os.path.join(WORK,f"cvec_{name}.gguf")
    w=GGUFWriter(path,"controlvector")
    w.add_string("controlvector.model_hint","gemma4")
    w.add_uint32("controlvector.layer_count",GL+1)
    w.add_tensor(f"direction.{GL+1}",vec)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"{path}: 1 layer (G-L{GL}), unit-norm")
