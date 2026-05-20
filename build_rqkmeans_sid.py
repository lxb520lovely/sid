#!/usr/bin/env python3
"""Build semantic IDs with PCA + residual-quantized K-Means.

Default configuration is tuned for the local Shopee title embedding file:

  input embedding: shopee_tittle_emb.npy, shape (N, 256)
  PCA/codeword dim: 64
  RQ levels: 4
  K-Means clusters per level: 512

Outputs are written to --output-dir:
  sid_codes.npy              int32, shape (N, L), values in [0, K)
  sid_codes_offset.npy       int32, shape (N, L), values in [level*K, ...)
  codebooks.npy              float32, shape (L, K, code_dim)
  projection.npz             PCA params used before RQ-KMeans
  sid_codeword_concat.npy    float32, shape (N, L*code_dim), optional dense feature
  sid_reconstruction.npy     float32, shape (N, code_dim), optional dense feature
  item_to_sid.csv            row item_id -> SID
  sid_to_items.json          SID string -> item ids
  metrics.json               utilization, entropy, collisions, reconstruction MSE
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct semantic IDs with PCA + RQ-KMeans."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("shopee_tittle_emb.npy"),
        help="Input .npy embedding matrix, shape (num_items, dim).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rqkmeans_sid_out"),
        help="Directory for SID artifacts.",
    )
    parser.add_argument(
        "--item-id-file",
        type=Path,
        default=None,
        help="Optional text file with one item id per line. Defaults to row index.",
    )
    parser.add_argument("--clusters", type=int, default=512, help="K per RQ level.")
    parser.add_argument("--levels", type=int, default=4, help="Number of RQ levels.")
    parser.add_argument(
        "--code-dim",
        type=int,
        default=64,
        help="PCA dimension before RQ-KMeans; also each codeword dimension.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=40,
        help="Maximum K-Means iterations per level.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Relative inertia improvement threshold for early stop.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Distance-computation batch size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--init",
        choices=("sample", "kmeans++"),
        default="sample",
        help="K-Means initialization. sample is faster; kmeans++ may improve quality.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="Number of K-Means restarts per level.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before PCA.",
    )
    parser.add_argument(
        "--no-dense-features",
        action="store_true",
        help="Do not save dense codeword concat/reconstruction features.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def load_item_ids(path: Path | None, n_items: int) -> list[str]:
    if path is None:
        return [str(i) for i in range(n_items)]

    ids = [line.rstrip("\n") for line in path.read_text().splitlines()]
    if len(ids) != n_items:
        raise ValueError(
            f"--item-id-file has {len(ids)} ids, but input has {n_items} rows"
        )
    if any(not item_id for item_id in ids):
        raise ValueError("--item-id-file contains empty item ids")
    return ids


def l2_normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
    return x / np.maximum(norms, EPS), norms.squeeze(1)


def pca_project(x: np.ndarray, code_dim: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if code_dim > x.shape[1]:
        raise ValueError(f"--code-dim={code_dim} exceeds input dim={x.shape[1]}")

    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = x - mean
    cov = (centered.T @ centered) / max(x.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)
    components = eigvecs[:, :code_dim].T.copy()
    projected = (centered @ components.T).astype(np.float32)

    total_var = float(np.maximum(eigvals.sum(), EPS))
    explained = eigvals[:code_dim]
    params = {
        "mean": mean,
        "components": components,
        "explained_variance": explained,
        "explained_variance_ratio": (explained / total_var).astype(np.float32),
        "all_explained_variance": eigvals,
    }
    return projected, params


def init_centers_sample(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if k > x.shape[0]:
        raise ValueError(f"clusters={k} exceeds num_items={x.shape[0]}")
    indices = rng.choice(x.shape[0], size=k, replace=False)
    return x[indices].copy()


def squared_distance_to_one_center(x: np.ndarray, center: np.ndarray) -> np.ndarray:
    diff = x - center[None, :]
    return np.einsum("ij,ij->i", diff, diff, optimize=True)


def init_centers_kmeans_pp(
    x: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    if k > x.shape[0]:
        raise ValueError(f"clusters={k} exceeds num_items={x.shape[0]}")

    n_items, dim = x.shape
    centers = np.empty((k, dim), dtype=np.float32)
    first_idx = int(rng.integers(0, n_items))
    centers[0] = x[first_idx]
    closest_sq = squared_distance_to_one_center(x, centers[0])

    for center_idx in range(1, k):
        total = float(closest_sq.sum())
        if not np.isfinite(total) or total <= EPS:
            next_idx = int(rng.integers(0, n_items))
        else:
            threshold = float(rng.random() * total)
            next_idx = int(np.searchsorted(np.cumsum(closest_sq), threshold))
            next_idx = min(next_idx, n_items - 1)
        centers[center_idx] = x[next_idx]
        new_dist = squared_distance_to_one_center(x, centers[center_idx])
        closest_sq = np.minimum(closest_sq, new_dist)

    return centers


def assign_to_centers(
    x: np.ndarray, centers: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    n_items = x.shape[0]
    labels = np.empty(n_items, dtype=np.int32)
    min_dist = np.empty(n_items, dtype=np.float32)
    center_norm = np.einsum("ij,ij->i", centers, centers, optimize=True)

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = x[start:end]
        batch_norm = np.einsum("ij,ij->i", batch, batch, optimize=True)
        dist = batch_norm[:, None] + center_norm[None, :] - 2.0 * (batch @ centers.T)
        dist = np.maximum(dist, 0.0)
        batch_labels = np.argmin(dist, axis=1)
        labels[start:end] = batch_labels.astype(np.int32)
        min_dist[start:end] = dist[np.arange(end - start), batch_labels].astype(
            np.float32
        )

    return labels, min_dist


def recompute_centers(
    x: np.ndarray,
    labels: np.ndarray,
    min_dist: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    dim = x.shape[1]
    centers = np.zeros((k, dim), dtype=np.float32)
    np.add.at(centers, labels, x)
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    non_empty = counts > 0
    centers[non_empty] /= counts[non_empty, None].astype(np.float32)

    empty = np.flatnonzero(~non_empty)
    if empty.size:
        farthest = np.argpartition(min_dist, -empty.size)[-empty.size:]
        farthest = farthest[np.argsort(min_dist[farthest])[::-1]]
        centers[empty] = x[farthest]

    return centers, counts


def run_kmeans_once(
    x: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    rng: np.random.Generator,
    init: str,
    verbose_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    if init == "kmeans++":
        centers = init_centers_kmeans_pp(x, k, rng)
    else:
        centers = init_centers_sample(x, k, rng)

    previous_inertia: float | None = None
    labels: np.ndarray | None = None
    min_dist: np.ndarray | None = None

    for iteration in range(1, max_iter + 1):
        labels, min_dist = assign_to_centers(x, centers, batch_size)
        inertia = float(min_dist.mean())
        centers, counts = recompute_centers(x, labels, min_dist, k)
        used = int(np.count_nonzero(counts))

        if previous_inertia is None:
            rel_improvement = math.inf
        else:
            rel_improvement = (previous_inertia - inertia) / max(previous_inertia, EPS)

        log(
            f"{verbose_prefix} iter={iteration:02d} "
            f"inertia={inertia:.8f} used={used}/{k} "
            f"rel_improve={rel_improvement:.6f}"
        )

        if previous_inertia is not None and rel_improvement >= 0 and rel_improvement < tol:
            break
        previous_inertia = inertia

    labels, min_dist = assign_to_centers(x, centers, batch_size)
    final_inertia = float(min_dist.mean())
    return centers, labels, min_dist, final_inertia, iteration


def run_kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    init: str,
    n_init: int,
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None = None
    for restart in range(n_init):
        rng = np.random.default_rng(seed + level * 1009 + restart * 9176)
        prefix = f"[level {level + 1} restart {restart + 1}/{n_init}]"
        result = run_kmeans_once(
            x=x,
            k=k,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            rng=rng,
            init=init,
            verbose_prefix=prefix,
        )
        if best is None or result[3] < best[3]:
            best = result
    assert best is not None
    return best


def entropy_from_counts(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    prob = positive / positive.sum()
    return float(-(prob * np.log2(prob)).sum())


def rq_kmeans(
    x: np.ndarray,
    k: int,
    levels: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    init: str,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    residual = x.copy()
    codebooks: list[np.ndarray] = []
    code_columns: list[np.ndarray] = []
    selected_codewords: list[np.ndarray] = []
    per_level: list[dict[str, Any]] = []

    for level in range(levels):
        log(f"Starting RQ level {level + 1}/{levels}")
        centers, labels, min_dist, inertia, used_iter = run_kmeans(
            x=residual,
            k=k,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            seed=seed,
            init=init,
            n_init=n_init,
            level=level,
        )

        counts = np.bincount(labels, minlength=k).astype(np.int64)
        selected = centers[labels]
        residual = (residual - selected).astype(np.float32)

        codebooks.append(centers.astype(np.float32))
        code_columns.append(labels.astype(np.int32))
        selected_codewords.append(selected.astype(np.float32))
        per_level.append(
            {
                "level": level,
                "inertia": inertia,
                "residual_mse_after_level": float(np.mean(np.sum(residual * residual, axis=1))),
                "iterations": used_iter,
                "used_codes": int(np.count_nonzero(counts)),
                "utilization": float(np.count_nonzero(counts) / k),
                "entropy": entropy_from_counts(counts),
                "min_cluster_size": int(counts[counts > 0].min()),
                "max_cluster_size": int(counts.max()),
                "counts": counts.tolist(),
            }
        )

    codes = np.stack(code_columns, axis=1).astype(np.int32)
    codebook_array = np.stack(codebooks, axis=0).astype(np.float32)
    selected_array = np.stack(selected_codewords, axis=1).astype(np.float32)
    metrics = {
        "per_level": per_level,
        "final_residual_mse": float(np.mean(np.sum(residual * residual, axis=1))),
    }
    return codes, codebook_array, metrics, selected_array


def collision_metrics(codes: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(codes)
    row_type = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    rows = contiguous.view(row_type).reshape(-1)
    _, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)

    collision_group_sizes = np.sort(counts[counts > 1])[::-1]
    return {
        "unique_sid": int(counts.shape[0]),
        "collision_groups": int(collision_group_sizes.shape[0]),
        "colliding_items": int(collision_group_sizes.sum()),
        "extra_collisions": int(codes.shape[0] - counts.shape[0]),
        "collision_rate_extra": float((codes.shape[0] - counts.shape[0]) / codes.shape[0]),
        "max_collision_group_size": int(collision_group_sizes[0]) if collision_group_sizes.size else 1,
        "top_collision_group_sizes": collision_group_sizes[:20].astype(int).tolist(),
        "inverse_group_id": inverse.astype(np.int32),
    }


def write_item_to_sid_csv(
    path: Path,
    item_ids: list[str],
    codes: np.ndarray,
    offset_codes: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_id", "item_id"]
        header.extend([f"sid_{i}" for i in range(codes.shape[1])])
        header.extend([f"sid_offset_{i}" for i in range(codes.shape[1])])
        header.append("sid")
        header.append("sid_offset")
        writer.writerow(header)
        for row_id, item_id in enumerate(item_ids):
            sid = codes[row_id].tolist()
            sid_offset = offset_codes[row_id].tolist()
            writer.writerow(
                [row_id, item_id]
                + sid
                + sid_offset
                + ["-".join(map(str, sid)), "-".join(map(str, sid_offset))]
            )


def write_sid_to_items_json(path: Path, item_ids: list[str], codes: np.ndarray) -> None:
    mapping: dict[str, list[str]] = {}
    for item_id, row in zip(item_ids, codes):
        key = "-".join(map(str, row.tolist()))
        mapping.setdefault(key, []).append(item_id)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading embeddings: {args.input}")
    raw = np.load(args.input)
    if raw.ndim != 2:
        raise ValueError(f"Input must be a 2-D matrix, got shape={raw.shape}")
    x = raw.astype(np.float32, copy=False)
    item_ids = load_item_ids(args.item_id_file, x.shape[0])

    raw_stats = {
        "shape": list(x.shape),
        "dtype": str(raw.dtype),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "nan_count": int(np.isnan(x).sum()),
        "inf_count": int(np.isinf(x).sum()),
    }
    if raw_stats["nan_count"] or raw_stats["inf_count"]:
        raise ValueError("Input contains NaN or Inf values")

    if args.no_normalize:
        normalized = x
        norms = np.linalg.norm(x, axis=1).astype(np.float32)
    else:
        log("Applying L2 normalization before PCA")
        normalized, norms = l2_normalize(x)

    log(f"Projecting embeddings to code_dim={args.code_dim} with PCA")
    projected, projection = pca_project(normalized, args.code_dim)
    explained_ratio_sum = float(projection["explained_variance_ratio"].sum())
    log(f"PCA explained variance ratio sum: {explained_ratio_sum:.6f}")

    log(
        "Running RQ-KMeans "
        f"K={args.clusters}, L={args.levels}, code_dim={args.code_dim}, init={args.init}"
    )
    codes, codebooks, rq_metrics, selected_codewords = rq_kmeans(
        x=projected,
        k=args.clusters,
        levels=args.levels,
        max_iter=args.max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
        seed=args.seed,
        init=args.init,
        n_init=args.n_init,
    )

    offsets = (np.arange(args.levels, dtype=np.int32) * args.clusters)[None, :]
    offset_codes = (codes + offsets).astype(np.int32)

    log("Computing collision metrics")
    collisions = collision_metrics(codes)
    inverse_group_id = collisions.pop("inverse_group_id")

    log("Writing artifacts")
    np.save(args.output_dir / "sid_codes.npy", codes)
    np.save(args.output_dir / "sid_codes_offset.npy", offset_codes)
    np.save(args.output_dir / "codebooks.npy", codebooks)
    np.save(args.output_dir / "sid_group_id.npy", inverse_group_id)
    np.savez_compressed(
        args.output_dir / "projection.npz",
        mean=projection["mean"],
        components=projection["components"],
        explained_variance=projection["explained_variance"],
        explained_variance_ratio=projection["explained_variance_ratio"],
        all_explained_variance=projection["all_explained_variance"],
        normalized=not args.no_normalize,
        input_norms=norms,
    )

    if not args.no_dense_features:
        concat_features = selected_codewords.reshape(
            selected_codewords.shape[0], args.levels * args.code_dim
        )
        reconstruction = selected_codewords.sum(axis=1)
        np.save(args.output_dir / "sid_codeword_concat.npy", concat_features.astype(np.float32))
        np.save(args.output_dir / "sid_reconstruction.npy", reconstruction.astype(np.float32))

    write_item_to_sid_csv(args.output_dir / "item_to_sid.csv", item_ids, codes, offset_codes)
    write_sid_to_items_json(args.output_dir / "sid_to_items.json", item_ids, codes)

    metrics = {
        "config": {
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "clusters": args.clusters,
            "levels": args.levels,
            "code_dim": args.code_dim,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "init": args.init,
            "n_init": args.n_init,
            "normalized": not args.no_normalize,
            "dense_features_saved": not args.no_dense_features,
        },
        "raw_embedding": raw_stats,
        "pca": {
            "projected_shape": list(projected.shape),
            "explained_variance_ratio_sum": explained_ratio_sum,
            "top_explained_variance_ratio": projection[
                "explained_variance_ratio"
            ][:10].astype(float).tolist(),
        },
        "rq_kmeans": rq_metrics,
        "sid": {
            "codes_shape": list(codes.shape),
            "codebook_shape": list(codebooks.shape),
            "offset_vocab_size": int(args.clusters * args.levels),
            "categorical_tokens_per_item": int(args.levels),
            "one_hot_sparse_dim_if_offset": int(args.clusters * args.levels),
            "dense_concat_dim_if_codewords": int(args.levels * args.code_dim),
            "dense_sum_dim_if_codewords": int(args.code_dim),
        },
        "collisions": collisions,
        "elapsed_seconds": float(time.time() - start_time),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log("Done")
    log(f"  sid_codes.npy: shape={codes.shape}")
    log(f"  codebooks.npy: shape={codebooks.shape}")
    log(f"  unique_sid={collisions['unique_sid']} / {codes.shape[0]}")
    log(
        "  extra_collisions="
        f"{collisions['extra_collisions']} "
        f"rate={collisions['collision_rate_extra']:.6f}"
    )
    log(f"  output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
