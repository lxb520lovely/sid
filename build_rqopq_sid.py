#!/usr/bin/env python3
"""Build semantic IDs with RQ-KMeans followed by OPQ.

Default configuration:

  input embedding: shopee_tittle_emb.npy, shape (N, 256)
  PCA/code dim: 32
  RQ:  K=512, L=3
  OPQ: M=2 subspaces, K=256 per subspace, subdim=16

The learned vectors are classical quantization centroids rather than neural
network parameters: RQ codewords are means of assigned residual vectors, and
OPQ codewords are means of assigned rotated residual subvectors.

Outputs are written to --output-dir:
  sid_codes.npy              int32, shape (N, RQ_L + OPQ_M)
  sid_codes_offset.npy       int32, same shape, safe for one shared vocabulary
  rq_codebooks.npy           float32, shape (RQ_L, RQ_K, code_dim)
  opq_codebooks.npy          float32, shape (OPQ_M, OPQ_K, subdim)
  opq_rotation.npy           float32, shape (code_dim, code_dim)
  projection.npz             PCA params used before quantization
  sid_codeword_concat.npy    float32, shape (N, RQ_L*code_dim + OPQ_M*subdim)
  sid_reconstruction.npy     float32, shape (N, code_dim)
  item_to_sid.csv            row item_id -> SID
  sid_to_items.json          SID string -> item ids
  metrics.json               utilization, entropy, collisions, reconstruction MSE
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from build_rqkmeans_sid import (
    EPS,
    collision_metrics,
    entropy_from_counts,
    l2_normalize,
    load_item_ids,
    log,
    pca_project,
    rq_kmeans,
    run_kmeans_once,
    write_sid_to_items_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct semantic IDs with PCA + RQ-KMeans + OPQ."
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
        default=Path("rqopq_sid_out"),
        help="Directory for SID artifacts.",
    )
    parser.add_argument(
        "--item-id-file",
        type=Path,
        default=None,
        help="Optional text file with one item id per line. Defaults to row index.",
    )
    parser.add_argument(
        "--code-dim",
        type=int,
        default=32,
        help="PCA dimension before quantization.",
    )
    parser.add_argument("--rq-clusters", type=int, default=512, help="RQ K.")
    parser.add_argument("--rq-levels", type=int, default=3, help="RQ levels.")
    parser.add_argument(
        "--opq-subspaces",
        type=int,
        default=2,
        help="Number of OPQ product subspaces.",
    )
    parser.add_argument(
        "--opq-clusters",
        type=int,
        default=256,
        help="K per OPQ subspace.",
    )
    parser.add_argument(
        "--rq-max-iter",
        type=int,
        default=35,
        help="Maximum K-Means iterations per RQ level.",
    )
    parser.add_argument(
        "--opq-outer-iter",
        type=int,
        default=4,
        help="OPQ alternating optimization rounds.",
    )
    parser.add_argument(
        "--opq-max-iter",
        type=int,
        default=30,
        help="Maximum K-Means iterations per OPQ subspace.",
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
        help="Number of K-Means restarts for each K-Means fit.",
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


def run_named_kmeans(
    x: np.ndarray,
    k: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    init: str,
    n_init: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None = None
    for restart in range(n_init):
        rng = np.random.default_rng(seed + restart * 9176)
        result = run_kmeans_once(
            x=x,
            k=k,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            rng=rng,
            init=init,
            verbose_prefix=f"[{name} restart {restart + 1}/{n_init}]",
        )
        if best is None or result[3] < best[3]:
            best = result
    assert best is not None
    return best


def solve_orthogonal_rotation(x: np.ndarray, quantized: np.ndarray) -> np.ndarray:
    """Solve min_R ||x R - quantized||_F with R constrained orthogonal."""
    cross = x.T @ quantized
    u, _, vt = np.linalg.svd(cross.astype(np.float64), full_matrices=False)
    return (u @ vt).astype(np.float32)


def train_product_quantizer(
    rotated: np.ndarray,
    subspaces: int,
    clusters: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    init: str,
    n_init: int,
    name_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n_items, dim = rotated.shape
    if dim % subspaces != 0:
        raise ValueError(
            f"code_dim={dim} must be divisible by opq_subspaces={subspaces}"
        )

    subdim = dim // subspaces
    codebooks: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    selected_parts: list[np.ndarray] = []
    min_dist_parts: list[np.ndarray] = []
    per_subspace: list[dict[str, Any]] = []

    for subspace in range(subspaces):
        start = subspace * subdim
        end = start + subdim
        part = rotated[:, start:end]
        log(f"Starting {name_prefix} OPQ subspace {subspace + 1}/{subspaces}")
        centers, labels, min_dist, inertia, used_iter = run_named_kmeans(
            x=part,
            k=clusters,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            seed=seed + subspace * 1009,
            init=init,
            n_init=n_init,
            name=f"{name_prefix} opq_subspace {subspace + 1}",
        )

        counts = np.bincount(labels, minlength=clusters).astype(np.int64)
        selected = centers[labels]

        codebooks.append(centers.astype(np.float32))
        codes.append(labels.astype(np.int32))
        selected_parts.append(selected.astype(np.float32))
        min_dist_parts.append(min_dist.astype(np.float32))
        per_subspace.append(
            {
                "subspace": subspace,
                "dim_range": [start, end],
                "inertia": float(inertia),
                "iterations": int(used_iter),
                "used_codes": int(np.count_nonzero(counts)),
                "utilization": float(np.count_nonzero(counts) / clusters),
                "entropy": entropy_from_counts(counts),
                "min_cluster_size": int(counts[counts > 0].min()),
                "max_cluster_size": int(counts.max()),
                "counts": counts.tolist(),
            }
        )

    selected_rotated = np.concatenate(selected_parts, axis=1).astype(np.float32)
    min_dist_matrix = np.stack(min_dist_parts, axis=1).astype(np.float32)
    return (
        np.stack(codebooks, axis=0).astype(np.float32),
        np.stack(codes, axis=1).astype(np.int32),
        selected_rotated,
        min_dist_matrix,
        per_subspace,
    )


def train_opq(
    residual: np.ndarray,
    subspaces: int,
    clusters: int,
    outer_iter: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    seed: int,
    init: str,
    n_init: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dim = residual.shape[1]
    if dim % subspaces != 0:
        raise ValueError(
            f"code_dim={dim} must be divisible by opq_subspaces={subspaces}"
        )
    if outer_iter < 1:
        raise ValueError("--opq-outer-iter must be at least 1")

    rotation = np.eye(dim, dtype=np.float32)
    history: list[dict[str, Any]] = []
    final_result: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ] | None = None

    for outer in range(outer_iter):
        log(f"Starting OPQ outer iteration {outer + 1}/{outer_iter}")
        rotated = (residual @ rotation).astype(np.float32)
        result = train_product_quantizer(
            rotated=rotated,
            subspaces=subspaces,
            clusters=clusters,
            max_iter=max_iter,
            tol=tol,
            batch_size=batch_size,
            seed=seed + outer * 100003,
            init=init,
            n_init=n_init,
            name_prefix=f"opq_outer {outer + 1}",
        )
        codebooks, codes, selected_rotated, min_dist_matrix, per_subspace = result

        rotated_mse = float(np.mean(min_dist_matrix.sum(axis=1)))
        reconstruction = selected_rotated @ rotation.T
        original_mse = float(np.mean(np.sum((residual - reconstruction) ** 2, axis=1)))
        history.append(
            {
                "outer_iter": outer,
                "rotated_mse": rotated_mse,
                "original_space_mse": original_mse,
                "per_subspace": per_subspace,
            }
        )
        log(
            f"[opq_outer {outer + 1}] "
            f"rotated_mse={rotated_mse:.8f} original_space_mse={original_mse:.8f}"
        )

        final_result = result
        if outer < outer_iter - 1:
            rotation = solve_orthogonal_rotation(residual, selected_rotated)

    assert final_result is not None
    codebooks, codes, selected_rotated, min_dist_matrix, per_subspace = final_result
    opq_reconstruction = (selected_rotated @ rotation.T).astype(np.float32)
    metrics = {
        "history": history,
        "final_rotated_mse": float(np.mean(min_dist_matrix.sum(axis=1))),
        "final_original_space_mse": float(
            np.mean(np.sum((residual - opq_reconstruction) ** 2, axis=1))
        ),
        "rotation_orthogonality_error": float(
            np.linalg.norm(rotation.T @ rotation - np.eye(dim, dtype=np.float32))
        ),
    }
    return codebooks, codes, selected_rotated, rotation, metrics


def make_offset_codes(
    rq_codes: np.ndarray,
    opq_codes: np.ndarray,
    rq_clusters: int,
    opq_clusters: int,
) -> np.ndarray:
    rq_offsets = np.arange(rq_codes.shape[1], dtype=np.int32) * rq_clusters
    opq_base = rq_codes.shape[1] * rq_clusters
    opq_offsets = opq_base + np.arange(opq_codes.shape[1], dtype=np.int32) * opq_clusters
    return np.concatenate(
        [
            rq_codes + rq_offsets[None, :],
            opq_codes + opq_offsets[None, :],
        ],
        axis=1,
    ).astype(np.int32)


def write_item_to_sid_csv(
    path: Path,
    item_ids: list[str],
    codes: np.ndarray,
    offset_codes: np.ndarray,
    rq_levels: int,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_id", "item_id"]
        header.extend([f"rq_sid_{i}" for i in range(rq_levels)])
        header.extend([f"opq_sid_{i}" for i in range(codes.shape[1] - rq_levels)])
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


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.code_dim % args.opq_subspaces != 0:
        raise ValueError(
            f"--code-dim={args.code_dim} must be divisible by "
            f"--opq-subspaces={args.opq_subspaces}"
        )

    subdim = args.code_dim // args.opq_subspaces
    log(
        "Using RQ-OPQ config: "
        f"RQ K={args.rq_clusters}, L={args.rq_levels}, code_dim={args.code_dim}; "
        f"OPQ M={args.opq_subspaces}, K={args.opq_clusters}, subdim={subdim}"
    )

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

    log("Running RQ-KMeans stage")
    rq_codes, rq_codebooks, rq_metrics, rq_selected = rq_kmeans(
        x=projected,
        k=args.rq_clusters,
        levels=args.rq_levels,
        max_iter=args.rq_max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
        seed=args.seed,
        init=args.init,
        n_init=args.n_init,
    )

    rq_reconstruction = rq_selected.sum(axis=1).astype(np.float32)
    residual = (projected - rq_reconstruction).astype(np.float32)
    log(
        "Residual after RQ stage: "
        f"mse={np.mean(np.sum(residual * residual, axis=1)):.8f}"
    )

    log("Running OPQ stage on final RQ residual")
    opq_codebooks, opq_codes, opq_selected_rotated, opq_rotation, opq_metrics = train_opq(
        residual=residual,
        subspaces=args.opq_subspaces,
        clusters=args.opq_clusters,
        outer_iter=args.opq_outer_iter,
        max_iter=args.opq_max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
        seed=args.seed + 1234567,
        init=args.init,
        n_init=args.n_init,
    )

    codes = np.concatenate([rq_codes, opq_codes], axis=1).astype(np.int32)
    offset_codes = make_offset_codes(
        rq_codes=rq_codes,
        opq_codes=opq_codes,
        rq_clusters=args.rq_clusters,
        opq_clusters=args.opq_clusters,
    )
    opq_reconstruction = (opq_selected_rotated @ opq_rotation.T).astype(np.float32)
    reconstruction = (rq_reconstruction + opq_reconstruction).astype(np.float32)

    log("Computing collision metrics")
    collisions = collision_metrics(codes)
    inverse_group_id = collisions.pop("inverse_group_id")

    log("Writing artifacts")
    np.save(args.output_dir / "sid_codes.npy", codes)
    np.save(args.output_dir / "sid_codes_offset.npy", offset_codes)
    np.save(args.output_dir / "rq_codebooks.npy", rq_codebooks)
    np.save(args.output_dir / "opq_codebooks.npy", opq_codebooks)
    np.save(args.output_dir / "opq_rotation.npy", opq_rotation)
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
        rq_concat = rq_selected.reshape(rq_selected.shape[0], -1)
        dense_concat = np.concatenate([rq_concat, opq_selected_rotated], axis=1)
        np.save(args.output_dir / "sid_codeword_concat.npy", dense_concat.astype(np.float32))
        np.save(args.output_dir / "sid_reconstruction.npy", reconstruction.astype(np.float32))
        np.save(
            args.output_dir / "opq_residual_reconstruction.npy",
            opq_reconstruction.astype(np.float32),
        )

    write_item_to_sid_csv(
        args.output_dir / "item_to_sid.csv",
        item_ids,
        codes,
        offset_codes,
        args.rq_levels,
    )
    write_sid_to_items_json(args.output_dir / "sid_to_items.json", item_ids, codes)

    offset_vocab_size = args.rq_levels * args.rq_clusters + args.opq_subspaces * args.opq_clusters
    metrics = {
        "config": {
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "code_dim": args.code_dim,
            "rq_clusters": args.rq_clusters,
            "rq_levels": args.rq_levels,
            "opq_subspaces": args.opq_subspaces,
            "opq_clusters": args.opq_clusters,
            "opq_subdim": subdim,
            "rq_max_iter": args.rq_max_iter,
            "opq_outer_iter": args.opq_outer_iter,
            "opq_max_iter": args.opq_max_iter,
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
        "opq": opq_metrics,
        "sid": {
            "codes_shape": list(codes.shape),
            "rq_codebook_shape": list(rq_codebooks.shape),
            "opq_codebook_shape": list(opq_codebooks.shape),
            "opq_rotation_shape": list(opq_rotation.shape),
            "offset_vocab_size": int(offset_vocab_size),
            "categorical_tokens_per_item": int(args.rq_levels + args.opq_subspaces),
            "one_hot_sparse_dim_if_offset": int(offset_vocab_size),
            "dense_concat_dim_if_codewords": int(
                args.rq_levels * args.code_dim + args.opq_subspaces * subdim
            ),
            "dense_reconstruction_dim": int(args.code_dim),
        },
        "reconstruction": {
            "rq_residual_mse": float(np.mean(np.sum(residual * residual, axis=1))),
            "opq_residual_mse": float(
                np.mean(np.sum((residual - opq_reconstruction) ** 2, axis=1))
            ),
            "total_projected_mse": float(
                np.mean(np.sum((projected - reconstruction) ** 2, axis=1))
            ),
        },
        "collisions": collisions,
        "elapsed_seconds": float(time.time() - start_time),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    log("Done")
    log(f"  sid_codes.npy: shape={codes.shape}")
    log(f"  rq_codebooks.npy: shape={rq_codebooks.shape}")
    log(f"  opq_codebooks.npy: shape={opq_codebooks.shape}")
    log(f"  opq_rotation.npy: shape={opq_rotation.shape}")
    log(f"  unique_sid={collisions['unique_sid']} / {codes.shape[0]}")
    log(
        "  extra_collisions="
        f"{collisions['extra_collisions']} "
        f"rate={collisions['collision_rate_extra']:.6f}"
    )
    log(f"  output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
