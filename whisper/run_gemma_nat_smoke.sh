#!/bin/bash
# Phase 3 A/B smoke: inherited boundary direction vs RANDOM control on gemma,
# same doses (64, 80) in the 48..96 window. Same pattern for both = noise
# verdict; boundary-flavoured output only for inherited = semantics survived.
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
OUT=work/gemma_nat_smoke.txt; : > $OUT
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT
systemctl --user stop qwen3.6-coding.service || true; sleep 3

for vec in boundary_gemma_native boundary_gemma; do
for dose in 40 64; do
  echo "=====VEC $vec DOSE $dose=====" >> $OUT
  $SRV -m $M --control-vector-scaled $PWD/work/cvec_$vec.gguf:$dose \
    --port 8091 -c 4096 -t 4 -tb 8 --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' > work/absmoke_srv.log 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 90); do curl -s -m 2 http://localhost:8091/health | grep -q ok && { ok=1; break; }; sleep 5; done
  if [ -z "$ok" ]; then echo "SERVER FAILED" >> $OUT; kill $SPID 2>/dev/null; SPID=""; continue; fi
  curl -s -m 400 http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" -d '{
    "messages":[{"role":"user","content":"Write a Python function last_n(lst, n) that returns the last n elements of a list. Reply with the complete function implementation in a single ```python code block."}],
    "temperature":0.0,"max_tokens":240}' | ~/interruptus-venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"][:800])' >> $OUT 2>&1
  kill $SPID 2>/dev/null; SPID=""; sleep 3
done
done
echo "NAT_DONE" >> $OUT
