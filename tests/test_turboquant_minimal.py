from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.turboquant_minimal import (
    QK_TQ3_0,
    TQ3Block,
    _pack_indices,
    _unpack_indices,
    dequantize_block,
    quantize_block,
)


def test_pack_roundtrip():
    indices = [i % 8 for i in range(QK_TQ3_0)]
    assert _unpack_indices(_pack_indices(indices)) == indices


def test_quantize_shape_and_gamma():
    block = quantize_block([float(i - 16) / 8.0 for i in range(QK_TQ3_0)])
    assert isinstance(block, TQ3Block)
    assert len(block.packed_indices) == 12
    assert block.gamma > 0


def test_zero_block_roundtrip_is_stable():
    restored = dequantize_block(quantize_block([0.0] * QK_TQ3_0))
    assert max(abs(v) for v in restored) < 1e-3


def test_signal_roundtrip_is_bounded():
    original = [((i % 7) - 3) / 3.0 for i in range(QK_TQ3_0)]
    restored = dequantize_block(quantize_block(original))
    mae = sum(abs(a - b) for a, b in zip(original, restored)) / QK_TQ3_0
    assert mae < 0.45


def test_sinusoid_roundtrip_is_bounded():
    import math

    original = [math.sin(i / 3.0) for i in range(QK_TQ3_0)]
    restored = dequantize_block(quantize_block(original))
    mae = sum(abs(a - b) for a, b in zip(original, restored)) / QK_TQ3_0
    assert mae < 0.35


def test_scale_invariance_is_reasonable():
    original = [((i % 5) - 2) * 0.25 for i in range(QK_TQ3_0)]
    scaled = [v * 4.0 for v in original]
    a = dequantize_block(quantize_block(original))
    b = dequantize_block(quantize_block(scaled))
    mae = sum(abs((x * 4.0) - y) for x, y in zip(a, b)) / QK_TQ3_0
    assert mae < 0.2
