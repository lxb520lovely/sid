from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

from .common import entropy_from_counts, iter_ranges, log
from .features import FusionBuilder


@dataclass
class RqResult:
    codebooks: list[np.ndarray]
    metrics: list[dict[str, Any]]


@dataclass
class OpqResult:
    codebooks: np.ndarray
    rotation: np.ndarray
    metrics_history: list[dict[str, Any]]


def nearest_centers(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x2 = np.einsum("ij,ij->i", x, x, optimize=True)
    c2 = np.einsum("ij,ij->i", centers, centers, optimize=True)
    dist = x2[:, None] + c2[None, :] - 2.0 * (x @ centers.T)
    dist = np.maximum(dist, 0.0)
    labels = np.argmin(dist, axis=1).astype(np.int32)
    min_dist = dist[np.arange(x.shape[0]), labels].astype(np.float32)
    return labels, min_dist


def compute_rq_residual(
    projected: np.ndarray,
    codes: np.ndarray,
    codebooks: list[np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    residual = np.asarray(projected[start:end], dtype=np.float32).copy()
    for level, centers in enumerate(codebooks):
        residual -= centers[np.asarray(codes[start:end, level], dtype=np.int32)]
    return residual


def fit_incremental_pca(
    fusion: FusionBuilder,
    n: int,
    code_dim: int,
    chunk_size: int,
    epochs: int,
) -> IncrementalPCA:
    ipca = IncrementalPCA(n_components=code_dim)
    for epoch in range(epochs):
        log(f"Fitting IncrementalPCA epoch {epoch + 1}/{epochs}")
        for start, end in iter_ranges(n, chunk_size):
            ipca.partial_fit(fusion.chunk(start, end))
    return ipca


def transform_projected(
    fusion: FusionBuilder,
    ipca: IncrementalPCA,
    out_path: Path,
    n: int,
    code_dim: int,
    chunk_size: int,
) -> np.ndarray:
    projected = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n, code_dim)
    )
    for start, end in iter_ranges(n, chunk_size):
        projected[start:end] = ipca.transform(fusion.chunk(start, end)).astype(np.float32)
    projected.flush()
    return projected


def fit_minibatch_kmeans(
    data_iter: Callable[[], Iterable[np.ndarray]],
    n_clusters: int,
    batch_size: int,
    epochs: int,
    seed: int,
    name: str,
) -> MiniBatchKMeans:
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init=1,
        reassignment_ratio=0.01,
        max_no_improvement=20,
        init_size=max(3 * n_clusters, batch_size),
    )
    for epoch in range(epochs):
        log(f"Training {name} epoch {epoch + 1}/{epochs}")
        for x in data_iter():
            km.partial_fit(x)
    return km


def summarize_counts(counts: np.ndarray) -> dict[str, Any]:
    nonzero = counts[counts > 0]
    return {
        "used_codes": int(nonzero.size),
        "utilization": float(nonzero.size / counts.size),
        "entropy": entropy_from_counts(counts),
        "min_cluster_size": int(nonzero.min()) if nonzero.size else 0,
        "max_cluster_size": int(counts.max()) if counts.size else 0,
        "counts": counts.astype(int).tolist(),
    }


def train_rq(
    projected: np.ndarray,
    codes: np.ndarray,
    n: int,
    rq_levels: int,
    rq_clusters: int,
    chunk_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> RqResult:
    codebooks: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    for level in range(rq_levels):
        log(f"Starting RQ level {level + 1}/{rq_levels}")

        def rq_train_iter(level=level):
            del level
            for start, end in iter_ranges(n, chunk_size):
                yield compute_rq_residual(projected, codes, codebooks, start, end)

        km = fit_minibatch_kmeans(
            data_iter=rq_train_iter,
            n_clusters=rq_clusters,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed + level * 1009,
            name=f"RQ level {level + 1}",
        )
        centers = km.cluster_centers_.astype(np.float32)
        codebooks.append(centers)

        counts = np.zeros(rq_clusters, dtype=np.int64)
        dist_sum = 0.0
        for start, end in iter_ranges(n, chunk_size):
            residual = compute_rq_residual(projected, codes, codebooks[:-1], start, end)
            labels, min_dist = nearest_centers(residual, centers)
            codes[start:end, level] = labels
            counts += np.bincount(labels, minlength=rq_clusters)
            dist_sum += float(min_dist.sum())
        metric = summarize_counts(counts)
        metric.update(
            {
                "level": level,
                "inertia": dist_sum / n,
                "residual_mse_after_level": dist_sum / n,
            }
        )
        metrics.append(metric)
        log(
            f"RQ level {level + 1}: mse={dist_sum/n:.8f}, "
            f"used={metric['used_codes']}/{rq_clusters}"
        )

    return RqResult(codebooks=codebooks, metrics=metrics)


def train_opq(
    projected: np.ndarray,
    codes: np.ndarray,
    rq_codebooks: list[np.ndarray],
    n: int,
    code_dim: int,
    rq_levels: int,
    opq_subspaces: int,
    opq_clusters: int,
    chunk_size: int,
    batch_size: int,
    outer_iter: int,
    epochs: int,
    seed: int,
) -> OpqResult:
    subdim = code_dim // opq_subspaces
    rotation = np.eye(code_dim, dtype=np.float32)
    opq_codebooks = np.zeros((opq_subspaces, opq_clusters, subdim), dtype=np.float32)
    metrics_history: list[dict[str, Any]] = []

    for outer in range(outer_iter):
        log(f"Starting OPQ outer iteration {outer + 1}/{outer_iter}")
        for subspace in range(opq_subspaces):
            s0, s1 = subspace * subdim, (subspace + 1) * subdim

            def opq_train_iter(subspace=subspace, s0=s0, s1=s1):
                del subspace
                for start, end in iter_ranges(n, chunk_size):
                    residual = compute_rq_residual(projected, codes, rq_codebooks, start, end)
                    rotated = residual @ rotation
                    yield rotated[:, s0:s1].astype(np.float32)

            km = fit_minibatch_kmeans(
                data_iter=opq_train_iter,
                n_clusters=opq_clusters,
                batch_size=batch_size,
                epochs=epochs,
                seed=seed + 1234567 + outer * 100003 + subspace * 1009,
                name=f"OPQ outer {outer + 1} subspace {subspace + 1}",
            )
            opq_codebooks[subspace] = km.cluster_centers_.astype(np.float32)

        counts_by_sub = [np.zeros(opq_clusters, dtype=np.int64) for _ in range(opq_subspaces)]
        min_dist_sum = np.zeros(opq_subspaces, dtype=np.float64)
        cross = np.zeros((code_dim, code_dim), dtype=np.float64)
        original_mse_sum = 0.0

        for start, end in iter_ranges(n, chunk_size):
            residual = compute_rq_residual(projected, codes, rq_codebooks, start, end)
            rotated = residual @ rotation
            selected_rotated = np.empty_like(rotated, dtype=np.float32)
            for subspace in range(opq_subspaces):
                s0, s1 = subspace * subdim, (subspace + 1) * subdim
                labels, min_dist = nearest_centers(rotated[:, s0:s1], opq_codebooks[subspace])
                codes[start:end, rq_levels + subspace] = labels
                selected_rotated[:, s0:s1] = opq_codebooks[subspace][labels]
                counts_by_sub[subspace] += np.bincount(labels, minlength=opq_clusters)
                min_dist_sum[subspace] += float(min_dist.sum())
            reconstruction = selected_rotated @ rotation.T
            original_mse_sum += float(np.sum((residual - reconstruction) ** 2))
            cross += residual.T @ selected_rotated

        per_subspace = []
        for subspace, counts in enumerate(counts_by_sub):
            metric = summarize_counts(counts)
            metric.update(
                {
                    "subspace": subspace,
                    "inertia": float(min_dist_sum[subspace] / n),
                    "dim_range": [subspace * subdim, (subspace + 1) * subdim],
                }
            )
            per_subspace.append(metric)

        original_mse = original_mse_sum / n
        metrics_history.append(
            {
                "outer_iter": outer,
                "rotated_mse": float(min_dist_sum.sum() / n),
                "original_space_mse": float(original_mse),
                "per_subspace": per_subspace,
            }
        )
        log(
            f"OPQ outer {outer + 1}: rotated_mse={min_dist_sum.sum()/n:.8f}, "
            f"original_mse={original_mse:.8f}"
        )

        if outer < outer_iter - 1:
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = (u @ vt).astype(np.float32)

    return OpqResult(
        codebooks=opq_codebooks,
        rotation=rotation,
        metrics_history=metrics_history,
    )


def make_offset_codes(
    codes: np.ndarray,
    rq_levels: int,
    rq_clusters: int,
    opq_clusters: int,
) -> np.ndarray:
    offsets = []
    for level in range(codes.shape[1]):
        if level < rq_levels:
            offsets.append(level * rq_clusters)
        else:
            offsets.append(rq_levels * rq_clusters + (level - rq_levels) * opq_clusters)
    return (codes + np.asarray(offsets, dtype=np.int32)[None, :]).astype(np.int32)


def write_offset_codes(
    out_path: Path,
    codes: np.ndarray,
    rq_levels: int,
    rq_clusters: int,
    opq_clusters: int,
    chunk_size: int,
) -> np.ndarray:
    offset_codes = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int32,
        shape=codes.shape,
    )
    for start, end in iter_ranges(codes.shape[0], chunk_size):
        offset_codes[start:end] = make_offset_codes(
            np.asarray(codes[start:end]),
            rq_levels,
            rq_clusters,
            opq_clusters,
        )
    offset_codes.flush()
    return offset_codes


def compute_collision_summary(codes: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(codes)
    row_type = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    rows = contiguous.view(row_type).reshape(-1)
    _, counts = np.unique(rows, return_counts=True)
    groups = np.sort(counts[counts > 1])[::-1]
    return {
        "unique_sid": int(counts.shape[0]),
        "unique_sid_ratio": float(counts.shape[0] / codes.shape[0]),
        "extra_collisions": int(codes.shape[0] - counts.shape[0]),
        "collision_rate_extra": float((codes.shape[0] - counts.shape[0]) / codes.shape[0]),
        "collision_groups": int(groups.shape[0]),
        "max_collision_group_size": int(groups[0]) if groups.size else 1,
        "top_collision_group_sizes": groups[:20].astype(int).tolist(),
    }
