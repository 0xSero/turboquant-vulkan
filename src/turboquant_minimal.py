from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence

QK_TQ3_0 = 32
TQ3_INV_SQRT_32 = 0.17677669529663688
TQ3_CENTROID_RMS = 1.3364
TQ3_CENTROIDS = (
    -2.1519, -1.3439, -0.7560, -0.2451,
    0.2451, 0.7560, 1.3439, 2.1519,
)
TQ3_BOUNDARIES = (
    -1.7479, -1.0500, -0.5006, 0.0,
    0.5006, 1.0500, 1.7479,
)
TQ3_SIGNS = (
    1,-1,1,1,-1,-1,1,-1, 1,1,-1,1,-1,1,-1,-1,
    1,-1,-1,1,1,-1,1,-1, -1,1,1,1,-1,-1,1,-1,
)

@dataclass(frozen=True)
class TQ3Block:
    packed_indices: bytes
    gamma: float

    def __post_init__(self) -> None:
        if len(self.packed_indices) != 12:
            raise ValueError('packed_indices must be 12 bytes for 32x3-bit values')
        if self.gamma <= 0:
            raise ValueError('gamma must be positive')


def _wht32(values: Sequence[float], *, inverse: bool) -> List[float]:
    if len(values) != QK_TQ3_0:
        raise ValueError('TurboQuant minimal reference expects 32 values')
    x = list(values)
    if not inverse:
        x = [v * s for v, s in zip(x, TQ3_SIGNS)]
    step = 1
    while step < QK_TQ3_0:
        for i in range(0, QK_TQ3_0, step * 2):
            for j in range(i, i + step):
                a = x[j]
                b = x[j + step]
                x[j] = a + b
                x[j + step] = a - b
        step <<= 1
    if inverse:
        return [v * TQ3_INV_SQRT_32 * s for v, s in zip(x, TQ3_SIGNS)]
    return [v * TQ3_INV_SQRT_32 for v in x]


def _quantize_index(value: float) -> int:
    idx = 0
    for boundary in TQ3_BOUNDARIES:
        if value > boundary:
            idx += 1
    return idx


def _pack_indices(indices: Sequence[int]) -> bytes:
    if len(indices) != QK_TQ3_0:
        raise ValueError('expected 32 indices')
    bits = 0
    for i, idx in enumerate(indices):
        if idx < 0 or idx > 7:
            raise ValueError('index out of range')
        bits |= idx << (i * 3)
    return bits.to_bytes(12, byteorder='little', signed=False)


def _unpack_indices(blob: bytes) -> List[int]:
    if len(blob) != 12:
        raise ValueError('packed index blob must be 12 bytes')
    bits = int.from_bytes(blob, byteorder='little', signed=False)
    return [(bits >> (i * 3)) & 0x7 for i in range(QK_TQ3_0)]


def quantize_block(values: Iterable[float]) -> TQ3Block:
    x = list(values)
    if len(x) != QK_TQ3_0:
        raise ValueError('expected exactly 32 values')
    rotated = _wht32(x, inverse=False)
    sum_sq = sum(v * v for v in rotated)
    gamma = sqrt(sum_sq / QK_TQ3_0) / TQ3_CENTROID_RMS
    if gamma <= 0:
        gamma = 1e-8
    inv_gamma = 1.0 / gamma
    indices = [_quantize_index(v * inv_gamma) for v in rotated]
    return TQ3Block(_pack_indices(indices), gamma)


def dequantize_block(block: TQ3Block) -> List[float]:
    rotated = [TQ3_CENTROIDS[idx] * block.gamma for idx in _unpack_indices(block.packed_indices)]
    return _wht32(rotated, inverse=True)
