#!/bin/bash
# Domain-matched anchors, clean version: render gemma's MBPP prompts (no gen),
# extract activations. Eval set (HumanEval) stays untouched -> zero leakage.
cd ~/interruptus
BIN=~/src/llama.cpp-latest/build/bin/llama-interruptus-extract
SRV=~/src/llama.cpp-latest/build/bin/llama-server
MODEL=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p work/feats_gemma_mbpp
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT

echo "$(date) stop coder"; systemctl --user stop qwen3.6-coding.service || true; sleep 3
echo "$(date) start gemma server (nur Template-Rendering)"
$SRV -m $MODEL --port 8088 -c 4096 -t 4 -tb 8 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' > work/server_gemma.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8088/health | grep -q ok && break; sleep 5; done
curl -s -m 2 http://localhost:8088/health | grep -q ok || { echo "server kam nicht hoch"; exit 1; }

echo "$(date) === render mbpp prompts ==="
~/interruptus-venv/bin/python render_mbpp_gemma.py > work/render_mbpp.log 2>&1 || { echo RENDER FAILED; tail -3 work/render_mbpp.log; exit 1; }
tail -1 work/render_mbpp.log

echo "$(date) kill server"; kill $SPID 2>/dev/null; SPID=""; sleep 5

echo "$(date) === extract gemma mbpp feats ==="
~/interruptus-venv/bin/python build_batch.py records_gemma_mbpp.jsonl batch_gemma_mbpp.txt
IEX_BATCH=$PWD/work/batch_gemma_mbpp.txt IEX_OUTDIR=$PWD/work/feats_gemma_mbpp \
  $BIN -m $MODEL -c 4096 -t 8 > work/ex_gemma_mbpp.log 2>&1
echo "$(date) done: $(ls work/feats_gemma_mbpp/*.f32 2>/dev/null | wc -l) feats"
echo "MBPP_DONE $(date)" > work/MBPP_DONE
