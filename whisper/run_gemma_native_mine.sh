#!/bin/bash
# Whisper research (a): mine the boundary direction NATIVELY on gemma —
# the inheritance ceiling. Render the 360 contrast prompts with gemma's own
# template, extract L29-36 last-token residuals, compute dirs, export cvec,
# then A/B smoke native-vs-inherited at the norm-calibrated dose.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
BIN=~/src/llama.cpp-latest/build/bin/llama-interruptus-extract
M=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p work/feats_whisper_gemma
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT
systemctl --user stop qwen3.6-coding.service || true; sleep 3

echo "$(date) server fuer Template-Rendering"
$SRV -m $M --port 8092 -c 4096 --parallel 1 -t 4 -tb 8 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' > work/srv_gnm.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8092/health | grep -q ok && break; sleep 5; done
curl -s -m 2 http://localhost:8092/health | grep -q ok || { echo "server kam nicht hoch"; exit 1; }

echo "$(date) render 360 Kontrast-Prompts im gemma-Format"
~/interruptus-venv/bin/python - <<'PYEOF'
import json, urllib.request, re, random, os
random.seed(23)   # SAME seed/recipe as build_whisper_corpus.py -> same tasks
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
rows=[json.loads(l) for l in open(os.path.expanduser("~/interruptus/data/mbpp.jsonl"))]
texts=[]; seen=set()
for r in rows:
    t=re.sub(r"\s+"," ",(r.get("text") or "").strip())
    if 6<=len(t.split())<=40 and t.lower() not in seen:
        seen.add(t.lower()); texts.append(t)
random.shuffle(texts); tasks=texts[:60]
def render(persona,task):
    msgs=[{"role":"user","content":f"{persona}\n\n{task} Reply with the complete function implementation in a single ```python code block."}]
    req=urllib.request.Request("http://localhost:8092/apply-template",
        data=json.dumps({"messages":msgs}).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as x: return json.load(x)["prompt"]
n=0
with open("work/batch_whisper_gemma.txt","wb") as bf, open("work/whisper_gemma_meta.jsonl","w") as mf:
    for c,(pos,neg) in CONCEPTS.items():
        for i,task in enumerate(tasks):
            for pole,persona in (("pos",pos),("neg",neg)):
                aid=f"{c}_{pole}_{i}"
                pb=render(persona,task).encode()
                bf.write(f"{aid}\t{len(pb)}\n".encode()); bf.write(pb); bf.write(b"\n")
                mf.write(json.dumps({"id":aid,"concept":c,"pole":pole,"task_idx":i})+"\n")
                n+=1
print(f"rendered {n}")
PYEOF
kill $SPID 2>/dev/null; SPID=""; sleep 5

echo "$(date) extract (gemma, default layers 29-36)"
IEX_BATCH=$PWD/work/batch_whisper_gemma.txt IEX_OUTDIR=$PWD/work/feats_whisper_gemma \
  $BIN -m $M -c 4096 -t 8 > work/ex_whisper_gemma.log 2>&1
echo "$(date) done: $(ls work/feats_whisper_gemma/*.f32 2>/dev/null | wc -l) feats"
echo "GNM_DONE $(date)" > work/GNM_DONE
