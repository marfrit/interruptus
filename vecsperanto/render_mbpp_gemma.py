#!/usr/bin/env python3
# Render-only: gemma's own prompt_str for all MBPP tasks (no generation).
# Produces records_gemma_mbpp.jsonl compatible with build_batch.py, ids
# identical to A's (mbpp_<task_id>) so the pairs join on id.
import json, os, hashlib, sys
sys.path.insert(0, os.path.expanduser("~/interruptus"))
import gen_and_label_gemma as G

WORK=os.path.expanduser("~/interruptus/work")
out=open(os.path.join(WORK,"records_gemma_mbpp.jsonl"),"w")
n=0
for t in G.mbpp_tasks():
    tid=f"mbpp_{t['task_id']}"
    ps=G.apply_template(G.mbpp_messages(t))
    ph=hashlib.sha256(ps.encode()).hexdigest()[:16]
    out.write(json.dumps({"id":tid,"prompt_str":ps,"prompt_hash":ph})+"\n")
    n+=1
    if n%100==0: print(n,flush=True)
out.close()
print(f"rendered {n} mbpp prompts")
