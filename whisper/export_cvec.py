#!/usr/bin/env python3
# Export whisper directions as llama.cpp control-vector GGUFs (repeng
# convention: tensors "direction.<hidden_state_idx>", 1-based = block_idx+1).
import os, sys, numpy as np
sys.path.insert(0,os.path.expanduser("~/src/llama.cpp-latest/gguf-py"))
from gguf import GGUFWriter

WORK=os.path.expanduser("~/interruptus/work")
z=np.load(os.path.join(WORK,"whisper_dirs.npz"))
LAYERS=[24,25,26,27,28,29,30]
concepts=sorted({k.rsplit("_L",1)[0] for k in z.files})
for c in concepts:
    path=os.path.join(WORK,f"cvec_{c}.gguf")
    w=GGUFWriter(path,"controlvector")
    w.add_string("controlvector.model_hint","qwen3moe")
    w.add_uint32("controlvector.layer_count",max(LAYERS)+1)
    for L in LAYERS:
        d=z[f"{c}_L{L}"].astype(np.float32)
        d=d/np.linalg.norm(d)          # unit norm; dose comes from --control-vector-scaled
        w.add_tensor(f"direction.{L+1}",d)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"{path}: {len(LAYERS)} layers, unit-norm")
