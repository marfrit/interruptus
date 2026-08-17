#!/usr/bin/env python3
# Native gemma whisper directions (ceiling for inheritance): diff-in-means on
# the gemma-rendered contrast prompts, separation check, export boundary cvec
# at G-L29 (single layer, comparable to the inherited one).
import os, sys, json, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,os.path.expanduser("~/src/llama.cpp-latest/gguf-py"))
from gguf import GGUFWriter

WORK=os.path.expanduser("~/interruptus/work")
FD=os.path.join(WORK,"feats_whisper_gemma")
LAYERS=[29,30,31,32,33,34,35,36]; D=3840; HOLD=12

meta=[json.loads(l) for l in open(os.path.join(WORK,"whisper_gemma_meta.jsonl"))]
def vec(aid,L):
    v=np.fromfile(os.path.join(FD,aid+".f32"),dtype=np.float32)
    off=LAYERS.index(L); return v[off*D:(off+1)*D]

dirs={}
for c in sorted({m["concept"] for m in meta}):
    ms=[m for m in meta if m["concept"]==c]
    tasks=sorted({m["task_idx"] for m in ms}); ho=set(tasks[-HOLD:]); tr=[t for t in tasks if t not in ho]
    for L in LAYERS:
        P=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="pos" and m["task_idx"] in tr])
        N=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="neg" and m["task_idx"] in tr])
        d=P.mean(0)-N.mean(0)
        Ph=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="pos" and m["task_idx"] in ho])
        Nh=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="neg" and m["task_idx"] in ho])
        auc=roc_auc_score(np.r_[np.ones(len(Ph)),np.zeros(len(Nh))],np.r_[Ph@d,Nh@d])
        dirs[(c,L)]=d
        if L in (29,32,36): print(f"{c} L{L}: sep-AUC {auc:.3f} |d| {np.linalg.norm(d):.1f}")

# cos(native, inherited) at L29 — how much of the true direction did the pivot deliver?
d_nat=dirs[("boundary",29)].astype(np.float64); d_nat/=np.linalg.norm(d_nat)
import struct
# load inherited from the gguf we built (or recompute quickly): read tensor from cvec_boundary_gemma.gguf via gguf reader
from gguf import GGUFReader
r=GGUFReader(os.path.join(WORK,"cvec_boundary_gemma.gguf"))
t=[x for x in r.tensors if x.name=="direction.30"][0]
d_inh=np.array(t.data,dtype=np.float32).astype(np.float64); d_inh/=np.linalg.norm(d_inh)
print(f"\ncos(native_L29, inherited_L29) = {float(d_nat@d_inh):+.3f}")

w=GGUFWriter(os.path.join(WORK,"cvec_boundary_gemma_native.gguf"),"controlvector")
w.add_string("controlvector.model_hint","gemma4")
w.add_uint32("controlvector.layer_count",30)
w.add_tensor("direction.30",(d_nat).astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
np.savez(os.path.join(WORK,"whisper_dirs_gemma.npz"),**{f"{c}_L{L}":d for (c,L),d in dirs.items()})
print("exported cvec_boundary_gemma_native.gguf (L29, unit-norm) + whisper_dirs_gemma.npz")
