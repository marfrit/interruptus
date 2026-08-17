#!/bin/bash
# Whisper measurement wrap-up: three arms on qwen's 27 baseline fails @0.2 —
# nonecare, naming, and a RANDOM 7-layer control (closes the control gap of
# the earlier boundary 3/27 claim).
cd ~/interruptus
SRV=~/src/llama.cpp-latest/build/bin/llama-server
M=~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf
export XDG_RUNTIME_DIR=/run/user/$(id -u)
cleanup(){ echo "$(date) cleanup"; [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 2;
  systemctl --user start qwen3.6-coding.service || true; }
trap cleanup EXIT

# random 7-layer control vector in qwen space (same format as the mined ones)
~/interruptus-venv/bin/python - <<'PYEOF'
import os, sys, numpy as np
sys.path.insert(0,os.path.expanduser("~/src/llama.cpp-latest/gguf-py"))
from gguf import GGUFWriter
rng=np.random.default_rng(77)
w=GGUFWriter("work/cvec_random_qwen.gguf","controlvector")
w.add_string("controlvector.model_hint","qwen3moe")
w.add_uint32("controlvector.layer_count",31)
for L in [24,25,26,27,28,29,30]:
    d=rng.standard_normal(2048).astype(np.float32); d/=np.linalg.norm(d)
    w.add_tensor(f"direction.{L+1}",d)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("cvec_random_qwen.gguf written (7 layers, unit-norm)")
PYEOF

systemctl --user stop qwen3.6-coding.service || true; sleep 3
for vec in nonecare naming random_qwen; do
  echo "$(date) === arm $vec @0.2 ==="
  $SRV -m $M --control-vector-scaled $PWD/work/cvec_$vec.gguf:0.2 \
    --port 8090 -c 4096 -t 4 -tb 8 --jinja \
    --chat-template-kwargs '{"enable_thinking":false}' > work/rest_srv_$vec.log 2>&1 &
  SPID=$!
  ok=""
  for i in $(seq 1 90); do curl -s -m 2 http://localhost:8090/health | grep -q ok && { ok=1; break; }; sleep 5; done
  [ -z "$ok" ] && { echo "server failed $vec"; kill $SPID 2>/dev/null; SPID=""; continue; }
  RESCUE_IDS=$PWD/work/rescue_ids.txt ~/interruptus-venv/bin/python gen_and_label_rescue.py \
    --family humaneval --tag _rest_$vec > work/gen_rest_$vec.log 2>&1
  echo "$(date) $vec done: $(wc -l < work/records_rest_$vec.jsonl 2>/dev/null) records"
  kill $SPID 2>/dev/null; SPID=""; sleep 3
done
echo "REST_DONE $(date)" > work/REST_DONE
