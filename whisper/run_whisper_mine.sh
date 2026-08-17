#!/bin/bash
# Fluester-WP Phase 1: extract contrastive activations on the 27B coder.
# Short run (~15 min); coder server stopped for RAM, trap-restarted.
cd ~/interruptus
BIN=~/src/llama.cpp-latest/build/bin/llama-interruptus-extract
export XDG_RUNTIME_DIR=/run/user/$(id -u)
mkdir -p work/feats_whisper
restart(){ echo "$(date) restart coder"; systemctl --user start qwen3.6-coding.service || true; }
trap restart EXIT
echo "$(date) stop coder"; systemctl --user stop qwen3.6-coding.service || true; sleep 3
IEX_BATCH=$PWD/work/batch_whisper.txt IEX_OUTDIR=$PWD/work/feats_whisper \
  $BIN -m ~/models/qwen36-27b-a3b-coder-Q4_K_M.gguf -c 4096 -t 8 > work/ex_whisper.log 2>&1
echo "$(date) done: $(ls work/feats_whisper/*.f32 2>/dev/null | wc -l) feats"
echo "WHISPER_DONE $(date)" > work/WHISPER_DONE
