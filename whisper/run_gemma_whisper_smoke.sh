#!/bin/bash
# Phase 3 smoke: inherited boundary direction on gemma, dose-finding.
# Single layer (G-L29) vs A's 7 layers -> expect higher dose scale.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
OUT=work/gemma_whisper_smoke.txt; : > $OUT
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT
systemctl --user stop qwen3.6-coding.service || true; sleep 3

for dose in 48 96 192; do
  echo "=====DOSE $dose=====" >> $OUT
  CV=""; [ "$dose" != "0" ] && CV="--control-vector-scaled $PWD/work/cvec_boundary_gemma.gguf:$dose"
  $SRV -m $M $CV --port 8091 -c 4096 -t 4 -tb 8 --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' > work/gsmoke_srv_$dose.log 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 90); do curl -s -m 2 http://localhost:8091/health | grep -q ok && { ok=1; break; }; sleep 5; done
  if [ -z "$ok" ]; then echo "SERVER FAILED dose=$dose" >> $OUT; tail -3 work/gsmoke_srv_$dose.log >> $OUT; kill $SPID 2>/dev/null; SPID=""; continue; fi
  curl -s -m 400 http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" -d '{
    "messages":[{"role":"user","content":"Write a Python function last_n(lst, n) that returns the last n elements of a list. Reply with the complete function implementation in a single ```python code block."}],
    "temperature":0.0,"max_tokens":260}' | ~/interruptus-venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' >> $OUT 2>&1
  kill $SPID 2>/dev/null; SPID=""; sleep 3
done
echo "GSMOKE_DONE" >> $OUT
