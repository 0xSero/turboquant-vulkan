from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.turboquant_minimal import QK_TQ3_0, dequantize_block, quantize_block


def make_blocks(num_blocks: int, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-2.0, 2.0) for _ in range(QK_TQ3_0)] for _ in range(num_blocks)]


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def run(num_blocks: int = 5000) -> dict:
    blocks = make_blocks(num_blocks)

    t0 = time.perf_counter()
    quantized = [quantize_block(block) for block in blocks]
    t1 = time.perf_counter()
    restored = [dequantize_block(block) for block in quantized]
    t2 = time.perf_counter()

    errors = [mae(a, b) for a, b in zip(blocks, restored)]
    return {
        'num_blocks': num_blocks,
        'block_size': QK_TQ3_0,
        'quantize_seconds': t1 - t0,
        'dequantize_seconds': t2 - t1,
        'quantize_blocks_per_sec': num_blocks / (t1 - t0),
        'dequantize_blocks_per_sec': num_blocks / (t2 - t1),
        'mean_mae': statistics.mean(errors),
        'p95_mae': statistics.quantiles(errors, n=20)[18],
        'compression_ratio_vs_fp16': (QK_TQ3_0 * 2) / 14,
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, sort_keys=True))
