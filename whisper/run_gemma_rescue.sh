#!/bin/bash
# Phase 3 finale: rescue run on gemma's 23 baseline fails at dose 56,
# TWO arms: inherited boundary direction vs random control (same dose).
# Rescue>0 for boundary with ~0 for random = causal cross-model whisper.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT

~/interruptus-venv/bin/python - <<'PYEOF'
import json
fails=[json.loads(l)["id"] for l in open("work/records_gemma.jsonl") if json.loads(l)["label"]==0]
open("work/gemma_rescue_ids.txt","w").write("\n".join(fails)+"\n")
print(f"{len(fails)} gemma rescue ids")
src=open("gen_and_label_gemma.py").read()   # SERVER 8088 -> 8091
src=src.replace('SERVER = "http://localhost:8088"','SERVER = "http://localhost:8091"')
anchor="        tasks = tasks[:args.limit]"
assert anchor in src
src=src.replace(anchor, anchor+"\n\n    _rf = os.environ.get(\"RESCUE_IDS\")\n    if _rf:\n        _keep = {l.strip() for l in open(_rf)}\n        tasks = [x for x in tasks if x[1] in _keep]")
open("gen_and_label_gemma_rescue.py","w").write(src)
print("runner patched")
PYEOF

systemctl --user stop qwen3.6-coding.service || true; sleep 3
for vec in boundary_gemma random_gemma; do
  echo "$(date) === arm $vec @56 ==="
  $SRV -m $M --control-vector-scaled $PWD/work/cvec_$vec.gguf:56 \
    --port 8091 -c 4096 -t 4 -tb 8 --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' > work/grescue_srv_$vec.log 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 90); do curl -s -m 2 http://localhost:8091/health | grep -q ok && { ok=1; break; }; sleep 5; done
  [ -z "$ok" ] && { echo "server failed $vec"; kill $SPID 2>/dev/null; SPID=""; continue; }
  RESCUE_IDS=$PWD/work/gemma_rescue_ids.txt ~/interruptus-venv/bin/python gen_and_label_gemma_rescue.py \
    --family humaneval --tag _grescue_$vec > work/gen_grescue_$vec.log 2>&1
  echo "$(date) $vec done: $(wc -l < work/records_grescue_$vec.jsonl 2>/dev/null) records"
  kill $SPID 2>/dev/null; SPID=""; sleep 3
done
echo "GRESCUE_DONE $(date)" > work/GRESCUE_DONE
