from __future__ import annotations

from typing import Iterator

import numpy as np


EPS = 1e-12


def log(msg: str) -> None:
    print(msg, flush=True)


def iter_ranges(n: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def entropy_from_counts(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if positive.size == 0:
        return 0.0
    p = positive / positive.sum()
    return float(-(p * np.log2(p)).sum())


def resize_dim(x: np.ndarray, dim: int) -> np.ndarray:
    """Deterministically resize feature dimension by truncating or zero-padding."""
    if x.shape[1] == dim:
        return x.astype(np.float32, copy=False)
    if x.shape[1] > dim:
        return x[:, :dim].astype(np.float32, copy=False)
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
    return out

