#!/bin/bash
# Fluester-WP Phase 2b: RESCUE run — only the 27 baseline-fail HumanEval tasks,
# boundary whisper at the no-think-calibrated dose 0.2. pass->fail risk on this
# subset is zero by construction; measured quantity = pure rescue rate.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT

# rescue set + patched runner
~/interruptus-venv/bin/python - <<'PYEOF'
import json
fails=[json.loads(l)["id"] for l in open("work/records.jsonl")
       if json.loads(l)["id"].startswith("HumanEval") and json.loads(l)["label"]==0]
open("work/rescue_ids.txt","w").write("\n".join(fails)+"\n")
print(f"{len(fails)} rescue ids")
src=open("gen_and_label.py").read()
src=src.replace('SERVER = "http://localhost:8085"','SERVER = "http://localhost:8090"')
src=src.replace('MODEL_ID = "qwen3.6-coding"','MODEL_ID = "qwen3.6-coding+boundary0.2"')
anchor="        tasks = tasks[:args.limit]"
assert anchor in src
src=src.replace(anchor, anchor+"\n\n    _rf = os.environ.get(\"RESCUE_IDS\")\n    if _rf:\n        _keep = {l.strip() for l in open(_rf)}\n        tasks = [x for x in tasks if x[1] in _keep]")
open("gen_and_label_rescue.py","w").write(src)
print("runner patched")
PYEOF

systemctl --user stop qwen3.6-coding.service || true; sleep 3
$SRV -m $M --port 8090 -c 4096 -t 4 -tb 8 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --control-vector-scaled $PWD/work/cvec_boundary.gguf:0.2 > work/srv_rescue.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8090/health | grep -q ok && break; sleep 5; done
curl -s -m 2 http://localhost:8090/health | grep -q ok || { echo "server kam nicht hoch"; exit 1; }

RESCUE_IDS=$PWD/work/rescue_ids.txt ~/interruptus-venv/bin/python gen_and_label_rescue.py \
  --family humaneval --tag _rescue02 > work/gen_rescue.log 2>&1
echo "$(date) done: $(wc -l < work/records_rescue02.jsonl 2>/dev/null) records"
echo "RESCUE_DONE $(date)" > work/RESCUE_DONE
