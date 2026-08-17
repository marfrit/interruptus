#!/usr/bin/env python3
# Fluester-WP: difference-in-means directions per concept/layer + held-out
# separation check (can go red: if personas don't separate, the concept is dud).
import os, json, numpy as np
from sklearn.metrics import roc_auc_score

WORK=os.path.expanduser("~/interruptus/work")
FD=os.path.join(WORK,"feats_whisper")
LAYERS=[24,25,26,27,28,29,30]; D=2048
HOLD=12  # tasks held out per concept for the separation check

meta=[json.loads(l) for l in open(os.path.join(WORK,"whisper_meta.jsonl"))]
def vec(aid,L):
    v=np.fromfile(os.path.join(FD,aid+".f32"),dtype=np.float32)
    off=LAYERS.index(L); return v[off*D:(off+1)*D]

out={}
print(f"{'concept':>10} {'layer':>5} {'sep-AUC(held-out)':>18} {'|dir|':>8}")
for c in sorted({m["concept"] for m in meta}):
    ms=[m for m in meta if m["concept"]==c]
    tasks=sorted({m["task_idx"] for m in ms})
    ho=set(tasks[-HOLD:]); tr=[t for t in tasks if t not in ho]
    for L in LAYERS:
        P=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="pos" and m["task_idx"] in tr])
        N=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="neg" and m["task_idx"] in tr])
        d=P.mean(0)-N.mean(0)
        Ph=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="pos" and m["task_idx"] in ho])
        Nh=np.stack([vec(m["id"],L) for m in ms if m["pole"]=="neg" and m["task_idx"] in ho])
        y=np.r_[np.ones(len(Ph)),np.zeros(len(Nh))]
        s=np.r_[Ph@d,Nh@d]
        auc=roc_auc_score(y,s)
        out[(c,L)]=d
        print(f"{c:>10} {L:>5} {auc:>18.3f} {np.linalg.norm(d):>8.1f}")

np.savez(os.path.join(WORK,"whisper_dirs.npz"),
         **{f"{c}_L{L}":d for (c,L),d in out.items()})
print("saved whisper_dirs.npz")
