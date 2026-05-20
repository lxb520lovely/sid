#!/usr/bin/env python3
"""Analyze a full multimodal RQ-OPQ SID run.

The builder already writes core training metrics. This script adds report-style
diagnostics that are cheap enough for 10M+ rows:

  * full-data prefix uniqueness and collision group distribution
  * sampled reconstruction cosine in PCA space
  * sampled nearest-neighbor prefix agreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a full SID output directory.")
    parser.add_argument("--sid-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--nn-sample-size", type=int, default=30000)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sid_to_str(row: np.ndarray) -> str:
    return "-".join(str(int(x)) for x in row)


def reconstruct(
    codes: np.ndarray,
    rq_codebooks: np.ndarray,
    opq_codebooks: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    rq_levels = rq_codebooks.shape[0]
    opq_subspaces = opq_codebooks.shape[0]
    subdim = opq_codebooks.shape[2]
    out = np.zeros((codes.shape[0], rq_codebooks.shape[2]), dtype=np.float32)
    for level in range(rq_levels):
        out += rq_codebooks[level, codes[:, level]]
    rotated_residual = np.zeros_like(out)
    for subspace in range(opq_subspaces):
        start = subspace * subdim
        end = start + subdim
        rotated_residual[:, start:end] = opq_codebooks[
            subspace, codes[:, rq_levels + subspace]
        ]
    out += rotated_residual @ rotation.T
    return out


def summarize_counts(counts: np.ndarray) -> dict[str, Any]:
    collision_counts = counts[counts > 1]
    if collision_counts.size == 0:
        return {
            "collision_groups": 0,
            "collision_items": 0,
            "max_group_size": 1,
            "quantiles": {},
        }
    quantile_points = [0.5, 0.9, 0.95, 0.99, 0.999]
    return {
        "collision_groups": int(collision_counts.size),
        "collision_items": int(collision_counts.sum()),
        "max_group_size": int(collision_counts.max()),
        "mean_collision_group_size": float(collision_counts.mean()),
        "quantiles": {
            str(q): float(np.quantile(collision_counts, q)) for q in quantile_points
        },
    }


def main() -> None:
    args = parse_args()
    sid_dir = args.sid_dir
    rng = np.random.default_rng(args.seed)

    codes = np.load(sid_dir / "sid_codes.npy", mmap_mode="r")
    projected = np.load(sid_dir / "projected.npy", mmap_mode="r")
    rq_codebooks = np.load(sid_dir / "rq_codebooks.npy")
    opq_codebooks = np.load(sid_dir / "opq_codebooks.npy")
    rotation = np.load(sid_dir / "opq_rotation.npy")

    n = codes.shape[0]
    prefix_stats = []
    final_counts = None
    final_unique = None
    for depth in range(1, codes.shape[1] + 1):
        unique_codes, counts = np.unique(np.asarray(codes[:, :depth]), axis=0, return_counts=True)
        if depth == codes.shape[1]:
            final_counts = counts
            final_unique = unique_codes
        prefix_stats.append(
            {
                "depth": depth,
                "unique": int(unique_codes.shape[0]),
                "unique_ratio": float(unique_codes.shape[0] / n),
                "extra_collision_rate": float((n - unique_codes.shape[0]) / n),
                "max_group_size": int(counts.max()),
            }
        )

    sample_size = min(args.sample_size, n)
    sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    sample_codes = np.asarray(codes[sample_idx])
    sample_projected = np.asarray(projected[sample_idx], dtype=np.float32)
    sample_recon = reconstruct(sample_codes, rq_codebooks, opq_codebooks, rotation)
    denom = np.maximum(
        np.linalg.norm(sample_projected, axis=1) * np.linalg.norm(sample_recon, axis=1),
        1e-12,
    )
    cosine = (sample_projected * sample_recon).sum(axis=1) / denom
    mse = ((sample_projected - sample_recon) ** 2).mean(axis=1)

    nn_n = min(args.nn_sample_size, n)
    nn_idx = np.sort(rng.choice(n, size=nn_n, replace=False))
    nn_codes = np.asarray(codes[nn_idx])
    nn_projected = np.asarray(projected[nn_idx], dtype=np.float32)
    neighbors = min(args.neighbors + 1, nn_n)
    nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute")
    nn.fit(nn_projected)
    _, indices = nn.kneighbors(nn_projected)
    neighbor_indices = indices[:, 1:]
    prefix_agreement = {}
    for depth in range(1, codes.shape[1] + 1):
        matches = (
            nn_codes[:, None, :depth] == nn_codes[neighbor_indices, :depth]
        ).all(axis=2)
        prefix_agreement[str(depth)] = float(matches.mean())

    assert final_counts is not None
    assert final_unique is not None
    order = np.argsort(-final_counts, kind="stable")[:20]
    top_groups = [
        {
            "sid": sid_to_str(final_unique[idx]),
            "group_size": int(final_counts[idx]),
        }
        for idx in order
    ]

    out = {
        "num_items": int(n),
        "prefix_stats": prefix_stats,
        "collision_distribution": summarize_counts(final_counts),
        "sampled_reconstruction": {
            "sample_size": int(sample_size),
            "per_dim_mse_mean": float(mse.mean()),
            "per_dim_mse_p50": float(np.quantile(mse, 0.5)),
            "per_dim_mse_p95": float(np.quantile(mse, 0.95)),
            "per_dim_mse_p99": float(np.quantile(mse, 0.99)),
            "vector_sse_mean": float(mse.mean() * projected.shape[1]),
            "cosine_mean": float(cosine.mean()),
            "cosine_p50": float(np.quantile(cosine, 0.5)),
            "cosine_p05": float(np.quantile(cosine, 0.05)),
            "cosine_p01": float(np.quantile(cosine, 0.01)),
        },
        "sampled_neighbor_prefix_agreement": {
            "sample_size": int(nn_n),
            "neighbors": int(neighbors - 1),
            "agreement_by_depth": prefix_agreement,
        },
        "top_collision_groups": top_groups,
    }

    output_path = sid_dir / "analysis_summary.json"
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
