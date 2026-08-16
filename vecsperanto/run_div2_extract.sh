#!/bin/bash
# Extract the larger diverse corpus (batch_div2.txt) through both models.
cd ~/interruptus
BIN=~/src/llama.cpp-latest/build/bin/llama-interruptus-extract
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p work/feats_div2_27b work/feats_div2_3b
restart_server(){ echo "$(date) restart coder server"; systemctl --user start qwen3.6-coding.service || true; }
trap restart_server EXIT
echo "$(date) stop coder server"; systemctl --user stop qwen3.6-coding.service || true; sleep 3
echo "$(date) === extract 27B ==="
IEX_BATCH=$PWD/work/batch_div2.txt IEX_OUTDIR=$PWD/work/feats_div2_27b \
  $BIN -m ~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf -c 4096 -t 8 > work/ex_div2_27b.log 2>&1
echo "$(date) 27B done: $(ls work/feats_div2_27b/*.f32 2>/dev/null | wc -l) feats"
echo "$(date) === extract 3B ==="
IEX_BATCH=$PWD/work/batch_div2.txt IEX_OUTDIR=$PWD/work/feats_div2_3b \
  $BIN -m ~/models/Qwen2.5-3B-Instruct-f16.gguf -c 4096 -t 8 > work/ex_div2_3b.log 2>&1
echo "$(date) 3B done: $(ls work/feats_div2_3b/*.f32 2>/dev/null | wc -l) feats"
echo "DIV2_DONE $(date)" > work/DIV2_DONE
