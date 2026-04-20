# TurboQuant Vulkan — Research References

## Primary Paper

**TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni
Google Research / NYU / Google DeepMind
ICLR 2026 | arXiv:2504.19874 (submitted April 2025)
https://arxiv.org/abs/2504.19874 | https://arxiv.org/html/2504.19874v1

### Key Results from Paper

- 3.5 bits/channel: "absolute quality neutrality" for KV cache
- 2.5 bits/channel: "marginal quality degradation"
- MSE distortion bound: D_mse <= (sqrt(3)*pi/2) * 1/4^b ≈ 2.72 * 1/4^b
- Only 2.7x from information-theoretic lower bound (Shannon)
- For b=1,2,3,4: D_mse ≈ 0.36, 0.117, 0.03, 0.009

### Algorithm Summary

**TurboQuant_MSE** (what we implement for KV cache):
1. Normalize input vector: x_hat = x / ||x||
2. Apply random rotation: x_rot = Q @ x_hat (WHT with fixed sign pattern)
3. Per-coordinate scalar quantization with Lloyd-Max codebook optimal for Beta(1/2, (d-1)/2)
4. Store packed indices + per-block scale

**TurboQuant_Prod** (for inner product / nearest neighbor):
1. Apply TurboQuant_MSE at b-1 bits
2. Apply 1-bit QJL transform on residual
3. Result: unbiased inner product estimator

The paper recommends Prod for attention keys (unbiased Q*K^T), but scos-lab experiments
show MSE-only beats Prod for perplexity (see Engineering Notes below).

### Theoretical Distortion (from paper Table 1)

| Bit-width b | D_mse (theory) | D_prod (theory) |
|-------------|----------------|-----------------|
| 1           | 0.360          | 1.570/d         |
| 2           | 0.117          | 0.560/d         |
| 3           | 0.030          | 0.180/d         |
| 4           | 0.009          | 0.047/d         |

### Information-Theoretic Lower Bound (Theorem 4)

Any quantizer Q: R^d → {0,1}^(b*d) must have:
- D_mse >= 1/4^b
- D_prod >= ||y||^2/(d * 4^b)

TurboQuant is within factor 2.72 of this bound (tight for small b).

---

## Reference Implementations

### scos-lab/turboquant (Python)
https://github.com/scos-lab/turboquant

Files: core.py, rotation.py, scalar_quantizer.py, qjl.py, mixed_precision.py,
kv_cache.py, compressed_cache.py, utils.py, tests/, benchmarks/

**Critical engineering findings** (not in paper):

1. **K/V Norm Disparity** — The hidden obstacle. Modern LLMs have K norms 6-1274x larger
   than V norms. Since quantization error scales with norm^2, K vectors dominate quality.
   Uniform bit allocation wastes bits on V.

   | Model | K/V Norm Ratio |
   |-------|---------------|
   | GPT-2 (124M) | 6x |
   | Phi-2 (2.8B) | 4x |
   | Qwen2.5-3B | 52x |
   | Qwen2.5-7B | 106x |
   | Qwen2.5-1.5B | 182x |
   | Qwen2.5-0.5B | 1274x |

2. **MSE > Prod for attention** — Contradicts paper recommendation.
   Tested: GPT-2 at b=4, MSE (both): +1.1% PPL, Paper (Prod keys): +6.5% PPL.
   Reason: softmax amplifies variance more than bias.

3. **Mixed precision at 3.6 bits** — Outlier-aware channel-level bit allocation.
   ~5-20% of K channels have 10-100x larger RMS. Store those at 8-bit, rest at 3-bit.
   Result on Qwen2.5-1.5B: +2.1% PPL at 3.6 avg bits (paper target: 3.5 bits, 0% PPL).

4. **K/V ratio predicts optimal budget**:
   - Ratio < 10x → 3-bit uniform works
   - Ratio 10-60x → 4.5-5 bit asymmetric
   - Ratio > 100x → 5.5+ bit or mixed precision

### AmesianX/TurboQuant (C++)
https://github.com/AmesianX/TurboQuant

C++ implementation for llama.cpp. Implements the core algorithm with
bit packing and WHT rotation.

### unixsysdev/llama-turboquant (C + HIP)
https://github.com/unixsysdev/llama-turboquant

AMD HIP implementation. Reports +5.8% PPL vs FP16 on tested models.

---

## Prior llama.cpp Integration Attempts

### PR #21010 — crayolaconsumer (Vulkan + CPU)
**Status:** Closed (AI policy violation + build failures)
**Branch:** pr/turboquant-tq3_0-vulkan
**What it did:** Added GGML_TYPE_TQ3_0 with Vulkan shaders and CPU reference

Key files changed (17 total):
- ggml.h, ggml-common.h, ggml.c, ggml-quants.{h,c}
- ggml-cpu/{ggml-cpu.c, quants.{h,c}, ops.cpp}
- ggml-vulkan/{ggml-vulkan.cpp, vulkan-shaders/*}
- common/arg.cpp

**Technical issues identified by reviewers:**

1. Used 4-level codebook {-1.510, -0.4528, +0.4528, +1.510} with 2-bit indices
   plus 1-bit QJL residual (stored but never used in dequant).
   Result: 2-bit quality at 3.5-bit storage cost.
   Fix: Use 8-level codebook with 3-bit indices (no wasted QJL bits).

2. Used `amax / 1.510` scaling (normalizes to max centroid magnitude).
   Fix: Use RMS-based scaling (minimizes MSE across entire block).

3. `vec_dot` set to NULL → CPU flash attention can't use TQ3_0.
   Fix: Implement vec_dot_tq3_0_q8_0.

4. No coopmat2 dequant function → shader compilation failures on NVIDIA.
   Fix: Either implement coopmat2 path or only register for scalar/coopmat1.

5. The `qs` array used `uint8_t[8]` (8 bytes for 32×2-bit) but should use
   12 bytes for 32×3-bit with the 8-level codebook.

**Vulkan shader approach (can be reused):**
- copy_to_quant.comp: WHT forward + codebook quantize, one thread per block
- dequant_funcs.glsl: WHT inverse + centroid lookup, returns vec4
- flash_attn_base.glsl: scaled dequant in FA inner loop
- types.glsl: block_tq3_0 + block_tq3_0_packed16 structs
- Pipeline registration: scalar + coopmat1 FA paths

### PR #21131 — peva3 (CUDA + CPU)
**Status:** Closed (AI policy violation)
**What it did:** Full TurboQuant with CUDA kernels, SIMD CPU path, H2O eviction

Key differences from #21010:
- Had CUDA kernels (turboquant_cuda.cu, .cuh, _streams.cpp)
- Had HIP support (turboquant_hip.cpp)
- Had AVX2/AVX-512 SIMD (turboquant_simd.cpp)
- Included H2O attention accumulator + hybrid eviction policy
- Much more invasive changes (modified llama-kv-cache.h heavily)

This PR was too complex and mixed too many concerns. The Vulkan-only approach
is more focused and aligns with our hardware.

### PR #21192 — crichalchemist (rotation only)
**Status:** Open
**What it does:** Adds just the TurboQuant random orthogonal rotation (Algorithm 1)
as a pre-transform before existing KV quantization types (q4_0, q8_0, etc.)

This is complementary: if merged, it could improve quality of existing quant types
without needing a new GGML type. But it doesn't achieve the full compression potential
of a dedicated TQ type.

### PR #20797 — 0cc4m (DP4A for quantized KV)
**Status:** Merged
**What it does:** Adds DP4A (integer dot product) shader to Vulkan FA for q8_0 and q4_0
KV cache types. This is the infrastructure we need for TQ3_0 FA — the pattern for
integrating a new KV type into the FA inner loop is established here.

---

## Vulkan Shader Architecture (from llama.cpp)

### Flash Attention Pipeline

Two paths exist:
1. **Scalar FA** (`flash_attn.comp`) — manual inner loop, no cooperative matrix
2. **Coopmat1 FA** (`flash_attn_cm1.comp`) — uses VK_KHR_cooperative_matrix

Both paths support KV types: f32, f16, q4_0, q8_0. Adding TQ3_0 follows the same pattern:
- Add `DATA_A_TQ3_0` define
- Implement `dequantFuncTQ3_0()` that returns vec4
- Register in vulkan-shaders-gen.cpp

### KV Quantize Pipeline

`copy_to_quant.comp` handles FP32→quant conversion on GPU.
Add a `#if defined(DATA_A_TQ3_0)` section with the WHT + codebook quantizer.
Workgroup size: 1 thread per 32-element block (same as other quant types).

### Type System

`types.glsl` defines block structs. Each quant type needs:
- `block_tq3_0` — standard layout (matches CPU struct)
- `block_tq3_0_packed16` — packed for efficient GPU loading

---

## QJL (Quantized Johnson-Lindenstrauss) Reference

From Zandieh et al. (2024), arXiv:2406.03482
Used in TurboQuant_Prod as the second stage (1-bit residual).

For our TQ3_0 implementation, we skip QJL entirely and use MSE-only reconstruction,
which scos-lab found gives better perplexity than Prod mode for attention workloads.

---

## Walsh-Hadamard Transform (WHT) Details

The WHT is used instead of a random rotation matrix Q because:
1. It's a fixed orthogonal matrix (no need to store Q)
2. O(n log n) instead of O(n^2) for matrix multiply
3. Self-inverse: WHT(WHT(x)) = x (up to normalization)
4. Data-oblivious: works without seeing the data
5. Induces near-Gaussian distribution on coordinates (CLT)

The sign preconditioning (tq3_signs) is a fixed random ±1 pattern that breaks
the structured correlation of the WHT. This is what makes it a "random rotation"
in the paper's sense — a randomized WHT is equivalent to a random orthogonal matrix
drawn from the Haar measure (approximately, for high dimensions).

WHT-32 butterfly schedule (5 stages):
```
step=1:  pairs (0,1), (2,3), (4,5), ..., (30,31)
step=2:  pairs (0,2), (1,3), (4,6), ..., (28,30)
step=4:  pairs (0,4), (1,5), (2,6), ..., (27,31)
step=8:  pairs (0,8), (1,9), (2,10), ..., (23,31)
step=16: pairs (0,16), (1,17), ..., (15,31)
```

Each pair: a,b → a+b, a-b. Total: 16*5 = 80 add/subtract operations.
Normalization: multiply all by 1/sqrt(32) = 0.17678.

This is fast in a GPU shader — all 32 values in registers, no shared memory needed.
