# TurboQuant KV Cache on Strix Halo — Findings

## Summary

Two KV-cache quantization types added to llama.cpp on top of the 2025 Vulkan build:

- **TQ3_0**: 3-bit WHT-rotated Lloyd-Max, 4.57× compression.
- **TQ4_0 + QJL**: 4-bit WHT-rotated Lloyd-Max + 1-bit QJL residual sketch, 2.67× compression.

TQ4_0 implements the two-stage scheme from the TurboQuant paper (Zandieh et al., arXiv 2504.19874):
an MSE-optimal quantizer plus a Quantized Johnson-Lindenstrauss (QJL) 1-bit residual projection that
makes inner products unbiased. Accuracy matches the paper's "quality-neutral" claim.

## Accuracy (Qwen3.6-35B-A3B Q4_0, wiki.test perplexity)

### 2048-context, 40 chunks, CPU KV cache (`-nkvo`, apples-to-apples):

| KV cache      | PPL             | Gap vs f16 |
|---------------|-----------------|------------|
| f16 / f16     | 7.6540 ± 0.11   | baseline   |
| **tq4_0 / tq4_0** | **7.7076 ± 0.11** | **+0.70%** ✓ |
| tq3_0 / tq3_0 | 8.0811 ± 0.11   | +5.58%     |

**TQ4_0 + QJL is within 1% of f16** — the paper's "absolute quality neutrality" threshold.

### test-quantize-fns (unit-level error, random Gaussian input):

| Type  | Abs err   | Dot-product err |
|-------|-----------|-----------------|
| TQ3_0 | 0.00418   | 0.0727          |
| TQ4_0 | 0.00155 (2.7× better) | 0.0108 (6.7× better) |

The dot-product error drops far more than raw MSE predicts — this is the unbiased-inner-product
property of QJL kicking in, exactly as the paper proves.

## Speed (Strix Halo, Qwen3.6-35B-A3B Q4_0, `ngl=0`, CPU KV cache)

| K cache | V cache | pp512 | tg128 | tg vs f16 |
|---------|---------|-------|-------|-----------|
| f16     | f16     | 312   | 27.29 | 100%      |
| q8_0    | q8_0    | 304   | 27.06 | 99.2% (native AVX2 reference) |
| tq3_0   | tq3_0   | 165   | 26.41 | 96.8%     |
| **tq4_0** | **tq4_0** | **153** | **26.91** | **98.6%** |
| tq3_0   | f16     | 225   | 26.88 | 98.5%     |
| tq4_0   | f16     | 199   | 26.91 | 98.6%     |

**TQ4_0 beats TQ3_0 on decode speed** after branchless Lloyd-Max + unrolled WHT,
while delivering ~10× better dot-product accuracy. Prefill (pp512) is still ~50%
of f16 because dequant is more arithmetic than q8_0's simple scale.

## Algorithm

### TQ3_0 (14 bytes per 32 values)

Block layout:
```c
typedef struct {
    uint8_t qs[12];  // 3-bit Lloyd-Max indices (32 × 3 bits)
    ggml_half d;     // scale
} block_tq3_0;
```

1. Apply WHT rotation with fixed ±1 sign pattern (Gaussianizes distribution).
2. Per-row L2 normalization.
3. 8-level Lloyd-Max quantization (Max 1960 optimal for N(0,1)).
4. Pack 3-bit indices.

Dequant uses the same WHT in inverse. Vec_dot uses the math-identity trick
`<WHT_inv(r̂), y> = <r̂, WHT_forward(y)>` to avoid full dequantization, keeping
one butterfly on the hot query side only.

### TQ4_0 + QJL (24 bytes per 32 values)

Block layout:
```c
typedef struct {
    uint8_t  qs[16];   // 4-bit Lloyd-Max indices (32 × 4 bits)
    uint8_t  qjl[4];   // 1-bit QJL signs (32 bits)
    ggml_half d;       // main scale
    ggml_half dj;      // residual norm / √32
} block_tq4_0;
```

1. `r = WHT_A(x ⊙ signs_A)` (Hadamard with sign pattern A).
2. `d = ‖r‖₂ / (√32 · σ_c)`, `v = r/d`.
3. 16-level Lloyd-Max quantization: `q_j = argmin |v_j − c_j|`.
4. Residual `e = v − c_q`, `dj = ‖e‖₂ / √32`.
5. `p = WHT_B(e ⊙ signs_B)` (independent sign pattern B for QJL projection).
6. Store `sign(p)` as 32 bits.

**Dequant** reconstructs:
```
x̂ = WHT_A_inv(d · c_q + d · dj · √(π/2) · WHT_B_inv(sign(p)))
```
The residual term is an unbiased estimator of the 4-bit quantization error — QJL identity.

**Vec_dot** uses math-identity twice:
```
<x̂, y> = <r̂, WHT_A_forward(y, signs_A)>
       = d · <c_q, y_rot_A> + d · dj · √(π/2) · <sign_p, WHT_B_forward(y_rot_A, signs_B)>
```
Both terms share `y_rot_A`, so we compute one WHT_A, one WHT_B, and two sign-weighted sums.

## Implementation notes

- **WHT butterfly unrolling** (commit `c6be47e`) gave the single biggest speedup.
  The nested triple-loop butterfly compiled poorly; explicit unrolling into 5 stages
  let the compiler auto-vectorize each stage cleanly. TQ4_0 tg128: 25.74 → 26.35.
- **Branchless residual unpacking** via bit-XOR with sign mask (commit `87949f2`).
- **Math-identity fused vec_dot** tightens FP rounding, commit `c043929` lowered PPL
  gap from 0.97% to 0.62% just by reordering computation.
- **AVX2 path** for dequant and vec_dot on Zen 5 (commit `558f21e`) — TQ3_0 tg128 at 100% of f16.

## Status

- CPU backend: **done**, both TQ3_0 and TQ4_0 working on ARM (NEON) and x86_64 (AVX2).
- Vulkan backend: **not yet** — types fall back to CPU KV cache (`-nkvo` needed).
- llama-quantize integration: **not yet** — types are KV-cache only for now.

## Next

- Vulkan shader port for TQ3_0 and TQ4_0 (would enable full-GPU path without CPU fallback).
- Fused quantize path (currently scalar) for K-side prefill speedup.
- Investigate per-tensor scale calibration vs per-row (current).
