# TurboQuant Vulkan — Implementation Plan

## Phase 0: CPU Reference (correctness first)

Implement the quantize/dequantize in pure C so we have a ground truth to test
the Vulkan shaders against.

### Files to create/modify in llama.cpp

```
ggml/include/ggml.h              — add GGML_TYPE_TQ3_0 enum
ggml/src/ggml-common.h           — add block_tq3_0 struct (14 bytes)
ggml/src/ggml-quants.h           — declare quantize/dequantize functions
ggml/src/ggml-quants.c           — implement WHT + 8-level codebook quantize/dequantize
ggml/src/ggml-cpu/quants.h       — declare quantize_row_tq3_0
ggml/src/ggml-cpu/quants.c       — implement quantize_row_tq3_0
ggml/src/ggml-cpu/ggml-cpu.c     — register type traits (from_float, vec_dot)
ggml/src/ggml-cpu/ops.cpp        — add TQ3_0 cases in switch statements
ggml/src/ggml.c                  — type_traits entry + quantize_tq3_0 in quantize_chunk
```

### Block layout

```c
#define QK_TQ3_0 32
typedef struct {
    uint16_t  qs[6];    // 12 bytes: 32 x 3-bit indices packed into 96 bits (6 x uint16)
    ggml_half gamma;    //  2 bytes: per-block scale = RMS of rotated vector / centroid RMS
} block_tq3_0;
// static_assert(sizeof(block_tq3_0) == 14, "wrong tq3_0 block size");
// 14 bytes / 32 elements = 3.5 bits/value = 4.57x compression vs FP16
```

### 8-level codebook (Lloyd-Max optimal for N(0,1), 3-bit)

```c
static const float tq3_centroids[8] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
    +0.2451f, +0.7560f, +1.3439f, +2.1519f
};

// Decision boundaries (midpoints between adjacent centroids)
static const float tq3_boundaries[7] = {
    -1.7479f, -1.0500f, -0.5006f, 0.0f,
    +0.5006f, +1.0500f, +1.7479f
};
```

### WHT rotation (self-inverse, in-register, no shared memory)

```c
static const int8_t tq3_signs[32] = {
    +1,-1,+1,+1,-1,-1,+1,-1, +1,+1,-1,+1,-1,+1,-1,-1,
    +1,-1,-1,+1,+1,-1,+1,-1, -1,+1,+1,+1,-1,-1,+1,-1
};
static const float TQ3_INV_SQRT_32 = 0.17677669529663688f; // 1/sqrt(32)

// Forward: signs → butterflies → normalize
static void tq3_wht32_forward(float *x) {
    for (int j = 0; j < 32; j++) x[j] *= tq3_signs[j];
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i += step * 2)
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
    for (int j = 0; j < 32; j++) x[j] *= TQ3_INV_SQRT_32;
}

// Inverse: butterflies → normalize AND undo signs (signs AFTER butterflies)
static void tq3_wht32_inverse(float *x) {
    for (int step = 1; step < 32; step <<= 1)
        for (int i = 0; i < 32; i += step * 2)
            for (int j = i; j < i + step; j++) {
                float a = x[j], b = x[j + step];
                x[j] = a + b; x[j + step] = a - b;
            }
    for (int j = 0; j < 32; j++) x[j] *= TQ3_INV_SQRT_32 * tq3_signs[j];
}
```

### Scaling: RMS-based (not amax-based)

```c
// gamma = sqrt(sum(x_rot[i]^2) / 32) / centroid_rms
// where centroid_rms = sqrt(sum(c[i]^2) / 8) for the 8-level codebook
static const float TQ3_CENTROID_RMS = 1.3364f; // sqrt(sum(ci^2)/8) for 8-level

float sum_sq = 0.0f;
for (int j = 0; j < 32; j++) sum_sq += rotated[j] * rotated[j];
float gamma = sqrtf(sum_sq / 32.0f) / TQ3_CENTROID_RMS;
block->gamma = GGML_FP32_TO_FP16(gamma);

// Quantize: idx = nearest centroid to rotated[j] / gamma
float inv_gamma = 1.0f / gamma;
for (int j = 0; j < 32; j++) {
    float xn = rotated[j] * inv_gamma;
    uint8_t idx = 0;
    // Binary search through 7 boundaries
    for (int b = 0; b < 7; b++)
        if (xn > tq3_boundaries[b]) idx = b + 1;
    // Pack 3-bit index into qs[]
    qs_packed[j * 3 / 16] |= (idx << (3 * j % 16));  // bit packing
}
```

### vec_dot for CPU flash attention

```c
// Dequantize a TQ3_0 block and compute dot product with a q8_0 vector
void vec_dot_tq3_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const block_tq3_0 * x = (const block_tq3_0 *)vx;
    const block_q8_0   * y = (const block_q8_0 *)vy;
    const int nb = n / QK_TQ3_0;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        // Dequantize TQ3_0 block
        float deq[32];
        dequantize_block_tq3_0(&x[i], deq);

        // Dot with Q8_0 block
        const float d8 = GGML_FP16_TO_FP32(y[i].d);
        float sumi = 0.0f;
        for (int j = 0; j < 32; j++) sumi += deq[j] * y[i].qs[j];
        sumf += sumi * d8;
    }
    *s = sumf;
}
```

---

## Phase 1: Vulkan Shaders

### 1a. types.glsl — Block struct

```glsl
struct block_tq3_0 {
    uint16_t qs[6];     // 32 x 3-bit packed
    float16_t gamma;    // per-block scale
};

struct block_tq3_0_packed16 {
    uint qs_lo;         // bits 0-31 (10.67 values)
    uint qs_mid;        // bits 32-63 (10.67 values)
    uint qs_hi;         // bits 64-95 (10.67 values)
    float16_t gamma;
};
```

### 1b. dequant_funcs.glsl — Inline dequant for mul_mat

```glsl
#if defined(DATA_A_TQ3_0)
const float tq3_centroids_d[8] = float[8](
    -2.1519, -1.3439, -0.7560, -0.2451,
    +0.2451, +0.7560, +1.3439, +2.1519
);
const int tq3_signs_d[32] = int[32](
    1,-1,1,1,-1,-1,1,-1, 1,1,-1,1,-1,1,-1,-1,
    1,-1,-1,1,1,-1,1,-1, -1,1,1,1,-1,-1,1,-1
);

vec4 dequant_tq3_0(uint ib, uint iqs, uint a_offset) {
    // Unpack 3-bit indices from the packed qs array
    // Do WHT inverse in registers (32-element butterfly)
    // Return 4 floats from position iqs..iqs+3
    // ... (full implementation in shader)
}
#endif
```

### 1c. copy_to_quant.comp — GPU quantizer

```glsl
// One workgroup per block (32 elements)
// Read 32 floats, apply WHT forward, quantize with 8-level codebook, write 14 bytes
void quantize(uint dst_idx, uint src_idx) {
    float rotated[32];
    // WHT forward
    for (uint j = 0; j < 32; j++)
        rotated[j] = float(data_s[src_idx + j]) * float(tq3_signs_q[j]);
    // Butterfly stages (unrolled)
    for (uint step = 1u; step < 32u; step <<= 1u)
        for (uint i = 0u; i < 32u; i += step * 2u)
            for (uint j = i; j < i + step; j++) {
                float a = rotated[j], b = rotated[j + step];
                rotated[j] = a + b; rotated[j + step] = a - b;
            }
    for (uint j = 0; j < 32; j++) rotated[j] *= 0.17677669529663688;

    // RMS scaling
    float sum_sq = 0.0;
    for (uint j = 0; j < 32; j++) sum_sq += rotated[j] * rotated[j];
    float gamma = sqrt(sum_sq / 32.0) / 1.3364;
    float inv_gamma = 1.0 / gamma;

    // Quantize with 8-level codebook
    uint qs_bits[3] = uint[3](0, 0, 0);
    for (uint j = 0; j < 32; j++) {
        float xn = rotated[j] * inv_gamma;
        uint idx = 0;
        if (xn > -1.7479) idx = 1;
        if (xn > -1.0500) idx = 2;
        if (xn > -0.5006) idx = 3;
        if (xn >  0.0000) idx = 4;
        if (xn >  0.5006) idx = 5;
        if (xn >  1.0500) idx = 6;
        if (xn >  1.7479) idx = 7;
        // Pack 3-bit index into 96-bit array
        uint bit_offset = j * 3u;
        qs_bits[bit_offset / 32u] |= (idx << (bit_offset % 32u));
    }
    // Write output
    data_q[dst_idx].qs[0] = qs_bits[0] & 0xFFFF;
    data_q[dst_idx].qs[1] = (qs_bits[0] >> 16) | ((qs_bits[1] & 0x3FF) << 16);  // etc.
    data_q[dst_idx].gamma = float16_t(gamma);
}
```

### 1d. flash_attn integration

For flash attention, TQ3_0 KV cache is dequantized inline during the Q*K^T and P*V matmuls.
Two paths:
- **Scalar FA**: dequant in the inner loop, feed float32 to the matmul
- **Coopmat1 FA**: dequant to shared memory (shmem), then coopmat loads from shmem

The coopmat1 path is more complex but ~2x faster on gfx1151. We implement scalar first,
then optimize to coopmat1.

---

## Phase 2: Validation

### Correctness

```bash
# Test 1: CPU quantize/dequantize round-trip
./llama-quantize --test-tq3_0

# Test 2: Vulkan shader matches CPU
./test-tq3-0-vulkan

# Test 3: Actual model inference with TQ3_0 KV cache
./llama-server -m qwen3.6-35b-q4km.gguf -ngl 99 -c 32768 \
    --cache-type-k tq3_0 --cache-type-v tq3_0 --flash-attn on
```

### Quality

```bash
# Perplexity on wikitext-2
./llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
    --cache-type-k tq3_0 --cache-type-v tq3_0 -c 8192

# Expected: <2% PPL increase vs FP16 KV at 3.5 bits
```

### Performance (on Strix Halo)

```bash
# Benchmark decode speed with TQ3_0 vs q4_0 and q8_0 KV
./llama-bench -m model.gguf -ngl 99 -c 8192,32768,131072 \
    --cache-type-k tq3_0,q4_0,q8_0 --cache-type-v tq3_0,q4_0,q8_0

# Expected: similar or slightly better tok/s than q4_0 (fewer bytes to transfer)
# Key metric: max context that fits in 128 GB with each KV type
```

---

## Phase 3: Optimizations (post-merge)

1. **Coopmat1 FA path** — dequant to shmem, load via coopmat. ~2x faster than scalar FA.
2. **TQ4_0** — 4.5-bit variant with 16-level codebook for even lower distortion.
3. **Mixed K/V precision** — K=TQ3_0, V=TQ2_0 to save more memory on V (which needs fewer bits).
4. **Outlier channel detection** — detect high-RMS channels, keep them at q8_0.

---

## Dependency Map

```
Phase 0 (CPU reference)
  ├── block struct + quantize/dequantize
  ├── vec_dot for CPU FA
  └── unit tests (round-trip, distortion)

Phase 1 (Vulkan shaders)  ← depends on Phase 0 for ground truth
  ├── types.glsl block struct
  ├── copy_to_quant.comp (GPU quantizer)
  ├── dequant_funcs.glsl (inline dequant for mul_mat)
  ├── flash_attn scalar path
  ├── ggml-vulkan.cpp pipeline registration
  └── shader compilation

Phase 2 (Validation)  ← depends on Phase 1
  ├── CPU round-trip test
  ├── Vulkan vs CPU match test
  ├── Perplexity benchmark
  ├── Decode speed benchmark on gfx1151
  └── Max context length measurement

Phase 3 (Optimizations)  ← depends on Phase 2 results
  ├── Coopmat1 FA path
  ├── TQ4_0 variant
  └── Mixed K/V precision
```
