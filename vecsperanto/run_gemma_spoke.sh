#!/bin/bash
# gemma-4-12b spoke chain, take 3: thinking disabled via the template's OWN
# enable_thinking gate (--chat-template-kwargs), not --reasoning-budget (which
# llama.cpp cannot enforce for gemma4's channel format — measured 2026-08-16).
# Includes a HARD-TASK format smoke test that ABORTS the chain if channel
# markers appear in output (no more hours of all-fail junk).
cd ~/interruptus
BIN=~/src/llama.cpp-latest/build/bin/llama-interruptus-extract
SRV=~/src/llama.cpp-latest/build/bin/llama-server
MODEL=~/models/gemma-4-12b-it-Q8_0.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p work/feats_gemmaown work/feats_div2_gemma
cleanup(){ echo "$(date) cleanup: kill gemma server, restart coder";
  [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT

echo "$(date) stop coder server"; systemctl --user stop qwen3.6-coding.service || true; sleep 3

echo "$(date) start gemma server :8088 (enable_thinking:false)"
$SRV -m $MODEL --port 8088 -c 4096 -t 4 -tb 8 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' > work/server_gemma.log 2>&1 &
SPID=$!
for i in $(seq 1 90); do curl -s -m 2 http://localhost:8088/health | grep -q ok && break; sleep 5; done
curl -s -m 2 http://localhost:8088/health | grep -q ok || { echo "server kam nicht hoch"; exit 1; }

echo "$(date) === format smoke test (real task) ==="
OUT=$(curl -s -m 300 http://localhost:8088/v1/chat/completions -H "Content-Type: application/json" -d '{
 "messages":[{"role":"user","content":"Complete the following Python function. Reply with the complete function implementation in a single ```python code block.\n\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.\"\"\""}],
 "max_tokens":400}')
CONTENT=$(printf '%s' "$OUT" | ~/interruptus-venv/bin/python -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null)
printf '%s' "$CONTENT" | head -c 200; echo
if printf '%s' "$CONTENT" | grep -q "channel\|<|"; then
  echo "SMOKE TEST FAILED: channel markers still present -- ABORT"; exit 1
fi
printf '%s' "$CONTENT" | grep -q '```python' || { echo "SMOKE TEST FAILED: no python block -- ABORT"; exit 1; }
echo "smoke test OK"

echo "$(date) === gen+label HumanEval on gemma ==="
~/interruptus-venv/bin/python gen_and_label_gemma.py --family humaneval --tag _gemma > work/gen_gemma.log 2>&1
echo "$(date) gen done: $(wc -l < work/records_gemma.jsonl 2>/dev/null) records"

echo "$(date) kill gemma server (RAM fuer Extraktor)"; kill $SPID 2>/dev/null; SPID=""; sleep 5

echo "$(date) === extract gemma own-prompt feats ==="
~/interruptus-venv/bin/python build_batch.py records_gemma.jsonl batch_gemmaown.txt
IEX_BATCH=$PWD/work/batch_gemmaown.txt IEX_OUTDIR=$PWD/work/feats_gemmaown \
  $BIN -m $MODEL -c 4096 -t 8 > work/ex_gemmaown.log 2>&1
echo "$(date) own done: $(ls work/feats_gemmaown/*.f32 2>/dev/null | wc -l) feats"

echo "$(date) === extract gemma div2 anchor feats ==="
IEX_BATCH=$PWD/work/batch_div2.txt IEX_OUTDIR=$PWD/work/feats_div2_gemma \
  $BIN -m $MODEL -c 4096 -t 8 > work/ex_div2_gemma.log 2>&1
echo "$(date) div2 done: $(ls work/feats_div2_gemma/*.f32 2>/dev/null | wc -l) feats"

echo "GEMMA_CHAIN_DONE $(date)" > work/GEMMA_DONE
