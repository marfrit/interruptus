#!/bin/bash
# Fluester-WP Phase 2: HumanEval pass-rate with boundary whisper at 0.5.
# Server config mirrors the baseline (:8085 recipe: --jinja + enable_thinking:false)
# so gen_and_label's rendered prompt_str matches records.jsonl byte-for-byte;
# the ONLY difference is the control vector.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT
systemctl --user stop qwen3.6-coding.service || true; sleep 3

$SRV -m $M --port 8090 -c 4096 -t 4 -tb 8 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --control-vector-scaled $PWD/work/cvec_boundary.gguf:0.5 > work/srv_cv05.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8090/health | grep -q ok && break; sleep 5; done
curl -s -m 2 http://localhost:8090/health | grep -q ok || { echo "server kam nicht hoch"; tail -5 work/srv_cv05.log; exit 1; }

sed -e "s#^SERVER = .*#SERVER = \"http://localhost:8090\"#" \
    -e "s#^MODEL_ID = .*#MODEL_ID = \"qwen3.6-coding+boundary0.5\"#" gen_and_label.py > gen_and_label_cv05.py
~/interruptus-venv/bin/python gen_and_label_cv05.py --family humaneval --tag _cv05 > work/gen_cv05.log 2>&1
echo "$(date) done: $(wc -l < work/records_cv05.jsonl 2>/dev/null) records"
echo "PHASE2_DONE $(date)" > work/PHASE2_DONE
