#!/usr/bin/env python3
# Fluester-WP Phase 1: contrastive prompt corpus for direction mining.
# 3 concepts x ~60 MBPP tasks x 2 poles = ~360 prompts, ChatML-wrapped to
# match the coder's serving template (differences cancel the template).
# Direction (later) = mean(act[pos]) - mean(act[neg]) per layer.
import json, os, random, re
WORK=os.path.expanduser("~/interruptus/work")
MBPP=os.path.expanduser("~/interruptus/data/mbpp.jsonl")
OUT=os.path.join(WORK,"batch_whisper.txt")
META=os.path.join(WORK,"whisper_meta.jsonl")
random.seed(23)

CONCEPTS={
 "boundary": (
  "You are an extremely careful programmer. Before writing any loop or index, you meticulously check boundary conditions: first element, last element, empty input, off-by-one in ranges.",
  "You are a fast, careless programmer. You write loops and indices from gut feeling and never think about boundary conditions or edge cases."),
 "naming": (
  "You are a programmer who chooses precise, descriptive variable and function names. Every identifier states exactly what it holds or does.",
  "You are a programmer who names everything x, y, tmp, foo and data. Names mean nothing to you and you never spend a second on them."),
 "nonecare": (
  "You are a defensive programmer. You always consider None values, missing keys, empty containers and invalid inputs before touching any data.",
  "You are a happy-path programmer. You assume every input is valid and present, and you never check for None, missing keys or empty containers."),
}

rows=[json.loads(l) for l in open(MBPP)]
texts=[]
seen=set()
for r in rows:
    t=re.sub(r"\s+"," ",(r.get("text") or "").strip())
    if 6<=len(t.split())<=40 and t.lower() not in seen:
        seen.add(t.lower()); texts.append(t)
random.shuffle(texts)
tasks=texts[:60]

def chatml(persona,task):
    return (f"<|im_start|>user\n{persona}\n\n{task} Reply with the complete "
            f"function implementation in a single ```python code block.<|im_end|>\n"
            f"<|im_start|>assistant\n")

n=0
with open(OUT,"wb") as bf, open(META,"w") as mf:
    for c,(pos,neg) in CONCEPTS.items():
        for i,task in enumerate(tasks):
            for pole,persona in (("pos",pos),("neg",neg)):
                aid=f"{c}_{pole}_{i}"
                pb=chatml(persona,task).encode()
                bf.write(f"{aid}\t{len(pb)}\n".encode()); bf.write(pb); bf.write(b"\n")
                mf.write(json.dumps({"id":aid,"concept":c,"pole":pole,"task_idx":i})+"\n")
                n+=1
print(f"wrote {n} prompts ({len(CONCEPTS)} concepts x {len(tasks)} tasks x 2 poles) -> {OUT}")
