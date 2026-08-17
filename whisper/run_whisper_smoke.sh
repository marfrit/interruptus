#!/bin/bash
# Whisper smoke test via llama-server (no llama-cli in this build).
# For each dose: fresh server with --control-vector-scaled, one greedy
# completion on a boundary-prone task WITHOUT persona, kill server.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
OUT=work/whisper_smoke.txt; : > $OUT
restart(){ echo "$(date) restart coder"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap restart EXIT
systemctl --user stop qwen3.6-coding.service || true; sleep 3

PROMPT='<|im_start|>user\nWrite a Python function last_n(lst, n) that returns the last n elements of a list. Reply with the complete function implementation in a single ```python code block.<|im_end|>\n<|im_start|>assistant\n'

for dose in 0.1 0.2 0.3; do
  echo "=====DOSE $dose=====" >> $OUT
  CV=""; [ "$dose" != "0" ] && CV="--control-vector-scaled $PWD/work/cvec_boundary.gguf:$dose"
  $SRV -m $M $CV --port 8090 -c 4096 -t 8 --jinja --chat-template-kwargs '{"enable_thinking":false}' > work/smoke_srv_$dose.log 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 60); do curl -s -m 2 http://localhost:8090/health | grep -q ok && { ok=1; break; }; sleep 5; done
  if [ -z "$ok" ]; then echo "SERVER FAILED dose=$dose" >> $OUT; tail -5 work/smoke_srv_$dose.log >> $OUT; kill $SPID 2>/dev/null; SPID=""; continue; fi
  ~/interruptus-venv/bin/python - "$OUT" <<'PYEOF'
import sys, json, urllib.request
prompt="<|im_start|>user\nWrite a Python function last_n(lst, n) that returns the last n elements of a list. Reply with the complete function implementation in a single ```python code block.<|im_end|>\n<|im_start|>assistant\n"
r=urllib.request.Request("http://localhost:8090/completion",
    data=json.dumps({"prompt":prompt,"temperature":0.0,"top_k":1,"n_predict":230,"cache_prompt":False,"stop":["<|im_end|>"]}).encode(),
    headers={"Content-Type":"application/json"})
with urllib.request.urlopen(r,timeout=600) as x:
    out=json.loads(x.read())["content"]
open(sys.argv[1],"a").write(out+"\n")
PYEOF
  kill $SPID 2>/dev/null; SPID=""; sleep 3
done
echo "SMOKE_DONE" >> $OUT
