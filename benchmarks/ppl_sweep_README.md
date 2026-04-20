# Perplexity Sweep — TQ3_0 vs Q4_0 vs F16 KV Cache

Context-length sweep measuring accuracy degradation (perplexity) when
swapping the KV cache quantization scheme.

## Methodology

- **Model:** Qwen3.5-0.8B-Q4_K_M (native 262144 context, no YaRN scaling)
- **Dataset:** WikiText-2 raw (train+test+valid concatenated, ~2.5M words)
- **Hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo), 128 GB LPDDR5X, CPU-only KV path
- **Ctx lengths:** 4K, 8K, 16K, 32K, 64K, 128K, 256K
- **KV configs:**
  - `f16/f16` (baseline)
  - `q4_0/q4_0` (4x K, 4x V compression)
  - `tq3_0/tq3_0` (4.57x K+V compression via WHT rotation + 8-level lattice)

## Why this matters

Decode speed is useless if accuracy collapses at long context. TQ3_0's
Walsh-Hadamard rotation is designed to make outlier-heavy attention
distributions much more Gaussian, giving better 3-bit quantization
headroom than naive scaling. This sweep tests whether that design holds
up at context lengths where naive KV quantization schemes usually fail.

## Results

See `benchmarks/results/ppl_sweep_qwen35_08b.jsonl` once complete.

Format per line:
```json
{"ctx": 4096, "kv_k": "tq3_0", "kv_v": "tq3_0", "ppl": 16.13, "elapsed_s": 14}
```

## Reproduce

```bash
# On Strix Halo (or any CPU with llama.cpp + tq3_0 patch applied)
./ppl_sweep.sh
```
