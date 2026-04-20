# TurboQuant Vulkan — Project Scope

## What This Is

A Vulkan compute shader implementation of TurboQuant KV cache compression for llama.cpp,
targeting the AMD Ryzen AI MAX+ 395 (gfx1151, RDNA 3.5, 128 GB UMA).

The goal: merge `GGML_TYPE_TQ3_0` (or an improved variant) into llama.cpp's Vulkan backend
so that Strix Halo users can run 2-4x longer context windows with near-zero quality loss.

## Why This Matters

On the Strix Halo, LLM decode is memory-bandwidth-bound at ~215 GB/s. A 70B Q4 model
(~35 GB weights) hits ~6 tok/s theoretical max. But at long context, the KV cache dominates:

| Context | KV cache (FP16, 70B) | KV cache (TQ3_0, 70B) | Reduction |
|---------|----------------------|------------------------|-----------|
| 8K      | 2.1 GB               | 0.46 GB                | 4.6x      |
| 32K     | 8.6 GB               | 1.87 GB                | 4.6x      |
| 128K    | 34.4 GB              | 7.5 GB                 | 4.6x      |

At 128K context, TQ3_0 saves 27 GB — enough room to run a 70B model that wouldn't
otherwise fit in 128 GB alongside weights + OS overhead.

## Prior Art (Why This Doesn't Exist Yet)

Three prior PRs attempted TurboQuant in llama.cpp. All were closed:

| PR | Author | Backend | Status | Why Closed |
|----|--------|---------|--------|------------|
| #21131 | peva3 | CUDA + CPU | Closed | AI policy violation |
| #21010 | crayolaconsumer | Vulkan + CPU | Closed | AI policy violation; also had build failures on cm2 path |
| #21307 | Keyvanhardani | ggml core | Closed | AI policy violation |
| #21192 | crichalchemist | CPU (Alg.1 rotation only) | Open | Just the rotation pre-transform, not full TQ |

Key technical issues from prior attempts:
1. **Codebook choice**: PR #21010 used a 4-level (2-bit) codebook + 1-bit QJL residual, but
   never used the QJL bits in dequant — effectively 2-bit quality at 3.5-bit cost.
   The paper's 3-bit MSE variant uses an 8-level codebook with lower distortion.
2. **Scaling**: PR #21010 used `amax / 1.510` (max centroid). The paper and reference impl
   use RMS-based scaling for better MSE.
3. **Coopmat2 path**: PR #21010 didn't implement the coopmat2 (NVIDIA-specific) dequant
   function, causing shader compilation failures.
4. **vec_dot missing**: No CPU vec_dot implementation, so CPU flash attention can't use TQ3_0.
5. **All were predominantly AI-generated**, violating llama.cpp contribution guidelines.

## What We Need to Build

### Core (required for a mergeable PR)

1. **CPU reference implementation** (`ggml-quants.c`)
   - `quantize_row_tq3_0_ref()` / `dequantize_row_tq3_0()`
   - Block struct: `block_tq3_0` in `ggml-common.h`
   - Type registration in `ggml.h`, `ggml.c`, `ggml-cpu/`
   - `vec_dot_tq3_0_q8_0()` for CPU flash attention

2. **Vulkan compute shaders**
   - `copy_to_quant.comp` — GPU quantizer (WHT + codebook + scaling)
   - `dequant_tq3_0.comp` — standalone dequant
   - `dequant_funcs.glsl` — inline dequant for mul_mat path
   - `flash_attn_base.glsl` — scaled dequant for flash attention inner loop
   - `types.glsl` — block struct + packed16 variant

3. **Vulkan backend integration** (`ggml-vulkan.cpp`)
   - Pipeline registration for scalar, coopmat1, (coopmat2 if needed for NVIDIA)
   - `supports_op()` allowlist entries
   - Shader compilation in `vulkan-shaders-gen.cpp`

4. **CLI integration** (`common/arg.cpp`)
   - Add `tq3_0` to `kv_cache_types`

### Validation (required before merge)

5. **Correctness tests**
   - Quantize/dequantize round-trip vs CPU reference
   - Bit-exact or within tolerance
   - All block sizes, edge cases

6. **Quality benchmarks**
   - Perplexity on wikitext-2 at various context lengths
   - Needle-in-a-haystack at 32K, 64K, 128K
   - Compare against q4_0, q8_0 KV baselines

7. **Performance benchmarks** (on gfx1151)
   - Decode tok/s with and without TQ3_0 at 8K, 32K, 128K context
   - Prefill tok/s impact
   - VRAM usage comparison

### Optional Enhancements

8. **TQ4_0 variant** — 4.5-bit with 16-level codebook, even lower distortion
9. **Mixed K/V precision** — K at TQ3_0, V at TQ2_0 (or vice versa)
10. **Outlier-aware channel allocation** — from scos-lab findings, 5-20% of K channels
    have 10-100x larger RMS; store those at 8-bit

## Algorithm Reference

### TurboQuant MSE (3-bit, what we implement)

Paper: Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate", ICLR 2026. arXiv:2504.19874

Algorithm (per block of 32 elements):

```
1. Normalize: x_norm = x / ||x||   (store ||x|| separately as float16)
2. Rotate:    x_rot = Q @ x_norm   (Q is a fixed random orthogonal matrix)
              In practice: Walsh-Hadamard Transform (WHT) with sign preconditioning
3. Quantize each coordinate with optimal Lloyd-Max codebook for Beta(1/2, (d-1)/2)
   For 3-bit (8 levels):
     centroids = {-2.1519, -1.3439, -0.7560, -0.2451, +0.2451, +0.7560, +1.3439, +2.1519}
4. Store: 3 bits × 32 elements = 12 bytes + 2 bytes scale = 14 bytes per block
```

Dequant reverses steps 3→2→1.

### Key Design Decisions (from research)

From scos-lab/turboquant reference implementation:

- **MSE mode for both K and V** (not Prod mode). The paper recommends Prod for keys,
  but experiments show softmax attention amplifies variance more than bias. MSE-only
  reconstruction gives better perplexity than unbiased Prod.
- **K/V norm disparity matters enormously**. Qwen models have K/V norm ratios of 50-1000x.
  Keys need more bits than values. Mixed precision (K=3, V=2) may be optimal.
- **WHT is self-inverse** — the forward and inverse transforms are identical
  (butterfly stages + normalization), only the sign preconditioning order differs.
  This simplifies the shader: one WHT function used both ways.

### Block Layout (TQ3_0 — 14 bytes per 32 elements)

```
struct block_tq3_0 {
    uint8_t   qs[12];    // 12 bytes: 32 x 3-bit codebook indices (bit-packed)
    ggml_half gamma;     //  2 bytes: per-block scale (norm)
};
// Total: 14 bytes for 32 values = 3.5 bits/value
// Compression vs FP16: 32 × 2 = 64 bytes → 14 bytes = 4.57x
```

Alternative 3-bit layout (8-level codebook, 12 bytes qs):

```
32 values × 3 bits = 96 bits = 12 bytes qs + 2 bytes gamma = 14 bytes total
```

## Target Hardware

| Property | Value |
|----------|-------|
| GPU | AMD Radeon 8060S (gfx1151, RDNA 3.5, 40 CU) |
| Memory | 128 GB LPDDR5X-8000 (~215 GB/s unified) |
| Vulkan | 1.3 with KHR_cooperative_matrix (coopmat1) |
| OS | Fedora 43, kernel 6.17+, mesa 26.x |
| llama.cpp | b8779 (Vulkan backend, flash attention supported) |

## References

### Paper
- Zandieh et al. (2025). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
  arXiv:2504.19874. ICLR 2026.
  https://arxiv.org/abs/2504.19874

### Reference Implementations
- scos-lab/turboquant — Python reference with engineering insights
  https://github.com/scos-lab/turboquant
  Key findings: MSE > Prod for attention, K/V norm disparity, mixed precision results

- unixsysdev/llama-turboquant — C + HIP reference
  https://github.com/unixsysdev/llama-turboquant

- AmesianX/TurboQuant — C++ implementation for llama.cpp
  https://github.com/AmesianX/TurboQuant

### Prior llama.cpp PRs
- #21131 (peva3) — CUDA + CPU, closed (AI policy)
- #21010 (crayolaconsumer) — Vulkan + CPU, closed (AI policy, build failures)
- #21307 (Keyvanhardani) — ggml core, closed (AI policy)
- #21192 (crichalchemist) — rotation-only, open
- #21131 review comments — Aaryan-Kapoor's 8-level codebook + vec_dot fork

### Related llama.cpp PRs (Vulkan FA + KV quant)
- #20797 (0cc4m) — DP4A shader for quantized KV cache in Vulkan FA
- #19075 (0cc4m) — Vulkan Flash Attention coopmat1 refactor
- #21029 (mkoker) — FA dequant for q4_1, q5_0, q5_1, iq4_nl
- #11166 (jeffbolznv) — Vulkan f32→quant copy support

### Vulkan Shader Reference
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp` — coopmat1 FA kernel
- `ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl` — inline dequant functions
- `ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp` — GPU quantizer template
- `ggml/src/ggml-vulkan/vulkan-shaders/types.glsl` — block struct definitions

### Engineering Notes
- scos-lab findings on K/V norm ratio: GPT-2 (6x), Qwen2.5-7B (106x), Qwen2.5-0.5B (1274x)
- 8-level codebook for N(0,1) via Lloyd-Max: {-2.1519, -1.3439, -0.7560, -0.2451, +0.2451, +0.7560, +1.3439, +2.1519}
- 4-level codebook: {-1.510, -0.4528, +0.4528, +1.510}
- WHT butterfly: `for (step=1; step<32; step<<=1) { a=x[j]; b=x[j+step]; x[j]=a+b; x[j+step]=a-b; }`
- WHT normalization: multiply by 1/sqrt(32) = 0.17678
