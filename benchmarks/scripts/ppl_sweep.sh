#!/bin/bash
# Perplexity sweep across context lengths and KV cache types
# Model: Qwen3.6-35B-A3B-Q4_0 on Strix Halo with Vulkan

set -u
cd ~/tq3_build/llama.cpp

MODEL=~/.local/share/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf
TEXT=~/tq3_build/wiki_long.raw
OUT=~/tq3_build/ppl_results.jsonl
LOG=~/tq3_build/ppl_sweep.log

: > "$OUT"
: > "$LOG"

# Context lengths to test. Ordered shortest→longest so we get early results.
CTXS=(4096 8192 16384 32768 65536 131072 262144)
# KV cache configs: (K, V)
CONFIGS=("f16 f16" "q4_0 q4_0" "tq3_0 tq3_0")

for ctx in "${CTXS[@]}"; do
    # Short ctx: more chunks for lower variance; long ctx: limited by dataset size
    if [ "$ctx" -le 8192 ]; then CHUNKS=8
    elif [ "$ctx" -le 32768 ]; then CHUNKS=4
    elif [ "$ctx" -le 131072 ]; then CHUNKS=2
    else CHUNKS=1
    fi
    for cfg in "${CONFIGS[@]}"; do
        read -r KTYPE VTYPE <<< "$cfg"
        echo "=== ctx=$ctx K=$KTYPE V=$VTYPE chunks=$CHUNKS ===" | tee -a "$LOG"
        START=$(date +%s)
        OUTPUT=$(LD_LIBRARY_PATH=$(pwd)/build/bin timeout 14400 ./build/bin/llama-perplexity \
            -m "$MODEL" \
            -f "$TEXT" \
            -c "$ctx" \
            -ngl 0 \
            -fa on \
            -ctk "$KTYPE" \
            -ctv "$VTYPE" \
            -t 16 \
            --chunks "$CHUNKS" \
            2>&1 | tee -a "$LOG")
        END=$(date +%s)
        ELAPSED=$((END-START))
        PPL=$(echo "$OUTPUT" | grep -oE 'Final estimate: PPL = [0-9.]+' | tail -1 | awk '{print $5}')
        PPL=${PPL:-"null"}
        printf '{"ctx":%d,"kv_k":"%s","kv_v":"%s","ppl":%s,"elapsed_s":%d}\n' \
            "$ctx" "$KTYPE" "$VTYPE" "$PPL" "$ELAPSED" | tee -a "$OUT"
        echo "=== done in ${ELAPSED}s, PPL=$PPL ===" | tee -a "$LOG"
    done
done

echo "SWEEP COMPLETE" | tee -a "$LOG"
