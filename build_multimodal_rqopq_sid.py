#!/usr/bin/env python3
"""Streaming multi-modal RQ-OPQ SID builder.

Baseline 4 input:
  title_emb.npy  -> dense title embedding, shape (N, 256)
  image_emb.npy  -> dense image embedding, shape (N, 256)
  item_feat.npz  -> itemid, cat_ids, flags, labels
  itemid.npy     -> raw item-id mapping for each row

The script builds a continuous multi-modal item representation with:
  L2(title) + L2(image) + discrete feature embeddings + flags

Discrete feature embeddings can be:
  random        -> deterministic random projection baseline
  semantic_mean -> streaming mean of title/image semantic base vectors

Then it runs:
  IncrementalPCA -> streaming RQ MiniBatchKMeans -> streaming OPQ MiniBatchKMeans

It is intentionally chunked/memmap based for 10M+ rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-modal RQ-OPQ SID.")
    parser.add_argument("--title-emb", type=Path, default=Path("title_emb.npy"))
    parser.add_argument("--image-emb", type=Path, default=Path("image_emb.npy"))
    parser.add_argument("--item-feat", type=Path, default=Path("item_feat.npz"))
    parser.add_argument("--itemid", type=Path, default=Path("itemid.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("multimodal_rqopq_sid_out"))

    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--rq-clusters", type=int, default=512)
    parser.add_argument("--rq-levels", type=int, default=3)
    parser.add_argument("--opq-subspaces", type=int, default=2)
    parser.add_argument("--opq-clusters", type=int, default=256)

    parser.add_argument("--cat-emb-dim", type=int, default=32)
    parser.add_argument("--label-emb-dim", type=int, default=32)
    parser.add_argument(
        "--discrete-embedding-mode",
        choices=("semantic_mean", "random"),
        default="semantic_mean",
        help="How to map cat_ids/labels to dense vectors.",
    )
    parser.add_argument("--title-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--cat-weight", type=float, default=0.5)
    parser.add_argument("--label-weight", type=float, default=0.5)
    parser.add_argument("--flag-weight", type=float, default=0.2)
    parser.add_argument(
        "--drop-constant-flags",
        action="store_true",
        default=True,
        help="Drop flag columns whose sampled/full min=max. Enabled by default.",
    )
    parser.add_argument(
        "--keep-constant-flags",
        action="store_false",
        dest="drop_constant_flags",
        help="Keep all flag columns.",
    )

    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--ipca-epochs", type=int, default=1)
    parser.add_argument("--rq-epochs", type=int, default=3)
    parser.add_argument("--opq-outer-iter", type=int, default=3)
    parser.add_argument("--opq-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Debug/smoke-test mode: only use the first N rows.",
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        help="Save sid_codeword_concat.npy and sid_reconstruction.npy. Large for full data.",
    )
    parser.add_argument(
        "--write-full-csv",
        action="store_true",
        help="Write full item_to_sid.csv. Large for full data.",
    )
    parser.add_argument(
        "--compute-collisions",
        action="store_true",
        help="Compute exact SID collisions with np.unique. Memory-heavy for full data.",
    )
    return parser.parse_args()


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


def make_random_table(vocab_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    table = rng.normal(0.0, 1.0 / math.sqrt(max(dim, 1)), size=(vocab_size, dim))
    table[0] *= 0.5
    return table.astype(np.float32)


def resize_dim(x: np.ndarray, dim: int) -> np.ndarray:
    """Deterministically resize feature dimension by truncating or zero-padding."""
    if x.shape[1] == dim:
        return x.astype(np.float32, copy=False)
    if x.shape[1] > dim:
        return x[:, :dim].astype(np.float32, copy=False)
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
    return out


def semantic_base_chunk(title: np.ndarray, image: np.ndarray, start: int, end: int) -> np.ndarray:
    """Base semantic vector used to aggregate discrete cat/label embeddings."""
    title_norm = l2_normalize(np.asarray(title[start:end], dtype=np.float32))
    image_norm = l2_normalize(np.asarray(image[start:end], dtype=np.float32))
    return ((title_norm + image_norm) * 0.5).astype(np.float32)


def build_semantic_mean_tables(
    title: np.ndarray,
    image: np.ndarray,
    feat: np.lib.npyio.NpzFile,
    n: int,
    cat_vocab_sizes: np.ndarray,
    label_vocab_size: int,
    cat_emb_dim: int,
    label_emb_dim: int,
    chunk_size: int,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Aggregate title/image semantic base vectors into cat/label tables."""
    base_dim = title.shape[1]
    cat_sums = [
        np.zeros((int(vocab), base_dim), dtype=np.float64)
        for vocab in cat_vocab_sizes.tolist()
    ]
    cat_counts = [
        np.zeros(int(vocab), dtype=np.int64) for vocab in cat_vocab_sizes.tolist()
    ]
    label_sum = np.zeros((label_vocab_size, base_dim), dtype=np.float64)
    label_count = np.zeros(label_vocab_size, dtype=np.int64)
    global_sum = np.zeros(base_dim, dtype=np.float64)
    global_count = 0

    for start, end in iter_ranges(n, chunk_size):
        base = semantic_base_chunk(title, image, start, end)
        cat_ids = np.asarray(feat["cat_ids"][start:end], dtype=np.int64)
        labels = np.asarray(feat["labels"][start:end], dtype=np.int64)
        global_sum += base.sum(axis=0, dtype=np.float64)
        global_count += end - start
        for field_idx in range(cat_ids.shape[1]):
            np.add.at(cat_sums[field_idx], cat_ids[:, field_idx], base)
            cat_counts[field_idx] += np.bincount(
                cat_ids[:, field_idx], minlength=cat_counts[field_idx].shape[0]
            )
        flat_labels = labels.reshape(-1)
        repeated_base = np.repeat(base, labels.shape[1], axis=0)
        np.add.at(label_sum, flat_labels, repeated_base)
        label_count += np.bincount(flat_labels, minlength=label_vocab_size)

    global_mean = global_sum / max(global_count, 1)
    cat_tables: list[np.ndarray] = []
    cat_stats = []
    for field_idx, (sums, counts) in enumerate(zip(cat_sums, cat_counts)):
        means = np.empty_like(sums, dtype=np.float32)
        seen = counts > 0
        means[seen] = (sums[seen] / counts[seen, None]).astype(np.float32)
        means[~seen] = global_mean.astype(np.float32)
        means = l2_normalize(means)
        means = resize_dim(means, cat_emb_dim)
        cat_tables.append(means)
        cat_stats.append(
            {
                "field": field_idx,
                "vocab_size": int(counts.shape[0]),
                "seen": int(seen.sum()),
                "unseen": int((~seen).sum()),
                "min_count_seen": int(counts[seen].min()) if seen.any() else 0,
                "max_count": int(counts.max()) if counts.size else 0,
            }
        )

    label_seen = label_count > 0
    label_means = np.empty_like(label_sum, dtype=np.float32)
    label_means[label_seen] = (
        label_sum[label_seen] / label_count[label_seen, None]
    ).astype(np.float32)
    label_means[~label_seen] = global_mean.astype(np.float32)
    label_means = resize_dim(l2_normalize(label_means), label_emb_dim)
    stats = {
        "semantic_base_dim": int(base_dim),
        "cat_stats": cat_stats,
        "label_stats": {
            "vocab_size": int(label_vocab_size),
            "seen": int(label_seen.sum()),
            "unseen": int((~label_seen).sum()),
            "min_count_seen": int(label_count[label_seen].min()) if label_seen.any() else 0,
            "max_count": int(label_count.max()) if label_count.size else 0,
        },
    }
    return cat_tables, label_means.astype(np.float32), stats


def validate_inputs(
    title: np.ndarray,
    image: np.ndarray,
    itemid: np.ndarray,
    feat: np.lib.npyio.NpzFile,
) -> int:
    n = title.shape[0]
    if title.ndim != 2 or image.ndim != 2:
        raise ValueError("title/image embeddings must be 2-D")
    if title.shape != image.shape:
        raise ValueError(f"title shape {title.shape} != image shape {image.shape}")
    if itemid.shape[0] != n:
        raise ValueError("itemid row count does not match embeddings")
    if feat["itemid"].shape[0] != n:
        raise ValueError("item_feat itemid row count does not match embeddings")
    if not np.array_equal(itemid[: min(n, 10000)], feat["itemid"][: min(n, 10000)]):
        raise ValueError("itemid.npy and item_feat.npz:itemid differ in first rows")
    for key in ["cat_ids", "flags", "labels"]:
        if feat[key].shape[0] != n:
            raise ValueError(f"item_feat {key} row count does not match")
    return n


def infer_active_flag_columns(flags: np.ndarray, n: int, chunk_size: int) -> list[int]:
    mins = np.full(flags.shape[1], np.inf, dtype=np.float64)
    maxs = np.full(flags.shape[1], -np.inf, dtype=np.float64)
    for start, end in iter_ranges(n, chunk_size):
        chunk = np.asarray(flags[start:end], dtype=np.float32)
        mins = np.minimum(mins, chunk.min(axis=0))
        maxs = np.maximum(maxs, chunk.max(axis=0))
    return [i for i, (lo, hi) in enumerate(zip(mins, maxs)) if lo != hi]


class FusionBuilder:
    def __init__(
        self,
        title: np.ndarray,
        image: np.ndarray,
        feat: np.lib.npyio.NpzFile,
        cat_tables: list[np.ndarray],
        label_table: np.ndarray,
        active_flag_cols: list[int],
        weights: dict[str, float],
    ) -> None:
        self.title = title
        self.image = image
        self.feat = feat
        self.cat_tables = cat_tables
        self.label_table = label_table
        self.active_flag_cols = active_flag_cols
        self.weights = weights

    @property
    def fused_dim(self) -> int:
        return (
            self.title.shape[1]
            + self.image.shape[1]
            + self.cat_tables[0].shape[1]
            + self.label_table.shape[1]
            + len(self.active_flag_cols)
        )

    def chunk(self, start: int, end: int) -> np.ndarray:
        title = l2_normalize(np.asarray(self.title[start:end], dtype=np.float32))
        image = l2_normalize(np.asarray(self.image[start:end], dtype=np.float32))
        cat_ids = np.asarray(self.feat["cat_ids"][start:end], dtype=np.int64)
        labels = np.asarray(self.feat["labels"][start:end], dtype=np.int64)
        flags = np.asarray(self.feat["flags"][start:end], dtype=np.float32)

        cat_emb = np.zeros((end - start, self.cat_tables[0].shape[1]), dtype=np.float32)
        for field_idx, table in enumerate(self.cat_tables):
            cat_emb += table[cat_ids[:, field_idx]]
        cat_emb /= math.sqrt(len(self.cat_tables))

        label_emb = self.label_table[labels].sum(axis=1).astype(np.float32)
        label_emb /= math.sqrt(labels.shape[1])

        active_flags = flags[:, self.active_flag_cols] if self.active_flag_cols else flags[:, :0]
        return np.concatenate(
            [
                title * self.weights["title"],
                image * self.weights["image"],
                cat_emb * self.weights["cat"],
                label_emb * self.weights["label"],
                active_flags * self.weights["flag"],
            ],
            axis=1,
        ).astype(np.float32)


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
    data_iter,
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


def write_item_to_sid_head(
    path: Path,
    itemid: np.ndarray,
    codes: np.ndarray,
    offset_codes: np.ndarray,
    n_rows: int = 10000,
) -> None:
    n = min(n_rows, codes.shape[0])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_id", "itemid"]
        header += [f"sid_{i}" for i in range(codes.shape[1])]
        header += [f"sid_offset_{i}" for i in range(codes.shape[1])]
        header += ["sid", "sid_offset"]
        writer.writerow(header)
        for row_id in range(n):
            sid = codes[row_id].tolist()
            sid_offset = offset_codes[row_id].tolist()
            writer.writerow(
                [row_id, int(itemid[row_id])]
                + sid
                + sid_offset
                + ["-".join(map(str, sid)), "-".join(map(str, sid_offset))]
            )


def write_item_to_sid_full(
    path: Path,
    itemid: np.ndarray,
    codes: np.ndarray,
    offset_codes: np.ndarray,
    chunk_size: int,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_id", "itemid"]
        header += [f"sid_{i}" for i in range(codes.shape[1])]
        header += [f"sid_offset_{i}" for i in range(codes.shape[1])]
        header += ["sid", "sid_offset"]
        writer.writerow(header)
        for start, end in iter_ranges(codes.shape[0], chunk_size):
            for row_id in range(start, end):
                sid = codes[row_id].tolist()
                sid_offset = offset_codes[row_id].tolist()
                writer.writerow(
                    [row_id, int(itemid[row_id])]
                    + sid
                    + sid_offset
                    + ["-".join(map(str, sid)), "-".join(map(str, sid_offset))]
                )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.code_dim % args.opq_subspaces != 0:
        raise ValueError("--code-dim must be divisible by --opq-subspaces")

    title = np.load(args.title_emb, mmap_mode="r", allow_pickle=False)
    image = np.load(args.image_emb, mmap_mode="r", allow_pickle=False)
    itemid = np.load(args.itemid, mmap_mode="r", allow_pickle=False)
    feat = np.load(args.item_feat, allow_pickle=False)
    n_total = validate_inputs(title, image, itemid, feat)
    n = min(n_total, args.max_items) if args.max_items else n_total

    log(f"Rows: using {n:,} / {n_total:,}")
    log(f"Title shape: {title.shape}, image shape: {image.shape}")

    cat_vocab_sizes = np.asarray(feat["cat_vocab_sizes"], dtype=np.int64)
    label_vocab_size = int(feat["label_vocab_size"][0])
    rng = np.random.default_rng(args.seed)
    discrete_table_stats: dict[str, Any]
    if args.discrete_embedding_mode == "random":
        log("Building random discrete embedding tables")
        cat_tables = [
            make_random_table(int(vocab), args.cat_emb_dim, rng)
            for vocab in cat_vocab_sizes.tolist()
        ]
        label_table = make_random_table(label_vocab_size, args.label_emb_dim, rng)
        discrete_table_stats = {"mode": "random"}
    else:
        log("Building semantic_mean discrete embedding tables")
        cat_tables, label_table, discrete_table_stats = build_semantic_mean_tables(
            title=title,
            image=image,
            feat=feat,
            n=n,
            cat_vocab_sizes=cat_vocab_sizes,
            label_vocab_size=label_vocab_size,
            cat_emb_dim=args.cat_emb_dim,
            label_emb_dim=args.label_emb_dim,
            chunk_size=args.chunk_size,
        )
        discrete_table_stats["mode"] = "semantic_mean"
    active_flag_cols = (
        infer_active_flag_columns(feat["flags"], n, args.chunk_size)
        if args.drop_constant_flags
        else list(range(feat["flags"].shape[1]))
    )
    log(f"Active flag columns: {active_flag_cols}")

    weights = {
        "title": args.title_weight,
        "image": args.image_weight,
        "cat": args.cat_weight,
        "label": args.label_weight,
        "flag": args.flag_weight,
    }
    fusion = FusionBuilder(
        title=title,
        image=image,
        feat=feat,
        cat_tables=cat_tables,
        label_table=label_table,
        active_flag_cols=active_flag_cols,
        weights=weights,
    )
    log(f"Fused feature dim before PCA: {fusion.fused_dim}")

    ipca = fit_incremental_pca(
        fusion=fusion,
        n=n,
        code_dim=args.code_dim,
        chunk_size=args.chunk_size,
        epochs=args.ipca_epochs,
    )
    projected = transform_projected(
        fusion=fusion,
        ipca=ipca,
        out_path=args.output_dir / "projected.npy",
        n=n,
        code_dim=args.code_dim,
        chunk_size=args.chunk_size,
    )

    codes = np.lib.format.open_memmap(
        args.output_dir / "sid_codes.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n, args.rq_levels + args.opq_subspaces),
    )

    rq_codebooks: list[np.ndarray] = []
    rq_metrics: list[dict[str, Any]] = []
    for level in range(args.rq_levels):
        log(f"Starting RQ level {level + 1}/{args.rq_levels}")

        def rq_train_iter(level=level):
            for start, end in iter_ranges(n, args.chunk_size):
                yield compute_rq_residual(projected, codes, rq_codebooks, start, end)

        km = fit_minibatch_kmeans(
            data_iter=rq_train_iter,
            n_clusters=args.rq_clusters,
            batch_size=args.batch_size,
            epochs=args.rq_epochs,
            seed=args.seed + level * 1009,
            name=f"RQ level {level + 1}",
        )
        centers = km.cluster_centers_.astype(np.float32)
        rq_codebooks.append(centers)

        counts = np.zeros(args.rq_clusters, dtype=np.int64)
        dist_sum = 0.0
        for start, end in iter_ranges(n, args.chunk_size):
            residual = compute_rq_residual(projected, codes, rq_codebooks[:-1], start, end)
            labels, min_dist = nearest_centers(residual, centers)
            codes[start:end, level] = labels
            counts += np.bincount(labels, minlength=args.rq_clusters)
            dist_sum += float(min_dist.sum())
        metric = summarize_counts(counts)
        metric.update(
            {
                "level": level,
                "inertia": dist_sum / n,
                "residual_mse_after_level": dist_sum / n,
            }
        )
        rq_metrics.append(metric)
        log(
            f"RQ level {level + 1}: mse={dist_sum/n:.8f}, "
            f"used={metric['used_codes']}/{args.rq_clusters}"
        )

    subdim = args.code_dim // args.opq_subspaces
    rotation = np.eye(args.code_dim, dtype=np.float32)
    opq_codebooks = np.zeros(
        (args.opq_subspaces, args.opq_clusters, subdim), dtype=np.float32
    )
    opq_metrics_history: list[dict[str, Any]] = []

    for outer in range(args.opq_outer_iter):
        log(f"Starting OPQ outer iteration {outer + 1}/{args.opq_outer_iter}")
        subspace_metrics = []
        for subspace in range(args.opq_subspaces):
            s0, s1 = subspace * subdim, (subspace + 1) * subdim

            def opq_train_iter(subspace=subspace, s0=s0, s1=s1):
                for start, end in iter_ranges(n, args.chunk_size):
                    residual = compute_rq_residual(projected, codes, rq_codebooks, start, end)
                    rotated = residual @ rotation
                    yield rotated[:, s0:s1].astype(np.float32)

            km = fit_minibatch_kmeans(
                data_iter=opq_train_iter,
                n_clusters=args.opq_clusters,
                batch_size=args.batch_size,
                epochs=args.opq_epochs,
                seed=args.seed + 1234567 + outer * 100003 + subspace * 1009,
                name=f"OPQ outer {outer + 1} subspace {subspace + 1}",
            )
            opq_codebooks[subspace] = km.cluster_centers_.astype(np.float32)

        counts_by_sub = [
            np.zeros(args.opq_clusters, dtype=np.int64)
            for _ in range(args.opq_subspaces)
        ]
        min_dist_sum = np.zeros(args.opq_subspaces, dtype=np.float64)
        cross = np.zeros((args.code_dim, args.code_dim), dtype=np.float64)
        original_mse_sum = 0.0

        for start, end in iter_ranges(n, args.chunk_size):
            residual = compute_rq_residual(projected, codes, rq_codebooks, start, end)
            rotated = residual @ rotation
            selected_rotated = np.empty_like(rotated, dtype=np.float32)
            for subspace in range(args.opq_subspaces):
                s0, s1 = subspace * subdim, (subspace + 1) * subdim
                labels, min_dist = nearest_centers(
                    rotated[:, s0:s1], opq_codebooks[subspace]
                )
                codes[start:end, args.rq_levels + subspace] = labels
                selected_rotated[:, s0:s1] = opq_codebooks[subspace][labels]
                counts_by_sub[subspace] += np.bincount(
                    labels, minlength=args.opq_clusters
                )
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
        opq_metrics_history.append(
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

        if outer < args.opq_outer_iter - 1:
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = (u @ vt).astype(np.float32)

    codes.flush()
    offset_codes = np.lib.format.open_memmap(
        args.output_dir / "sid_codes_offset.npy",
        mode="w+",
        dtype=np.int32,
        shape=codes.shape,
    )
    for start, end in iter_ranges(n, args.chunk_size):
        offset_codes[start:end] = make_offset_codes(
            np.asarray(codes[start:end]),
            args.rq_levels,
            args.rq_clusters,
            args.opq_clusters,
        )
    offset_codes.flush()

    np.save(args.output_dir / "rq_codebooks.npy", np.stack(rq_codebooks).astype(np.float32))
    np.save(args.output_dir / "opq_codebooks.npy", opq_codebooks.astype(np.float32))
    np.save(args.output_dir / "opq_rotation.npy", rotation.astype(np.float32))
    np.save(args.output_dir / "itemid.npy", np.asarray(itemid[:n], dtype=np.int64))
    np.savez_compressed(
        args.output_dir / "projection.npz",
        mean=ipca.mean_.astype(np.float32),
        components=ipca.components_.astype(np.float32),
        explained_variance=ipca.explained_variance_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        input_dim=np.asarray([fusion.fused_dim], dtype=np.int32),
    )
    np.savez_compressed(
        args.output_dir / "fusion_tables.npz",
        **{f"cat_table_{i}": table for i, table in enumerate(cat_tables)},
        label_table=label_table,
        active_flag_cols=np.asarray(active_flag_cols, dtype=np.int32),
        cat_vocab_sizes=cat_vocab_sizes.astype(np.int64),
        label_vocab_size=np.asarray([label_vocab_size], dtype=np.int64),
        discrete_embedding_mode=np.asarray([args.discrete_embedding_mode]),
        weights=np.asarray(
            [weights["title"], weights["image"], weights["cat"], weights["label"], weights["flag"]],
            dtype=np.float32,
        ),
    )

    if args.save_dense:
        dense_dim = args.rq_levels * args.code_dim + args.opq_subspaces * subdim
        dense = np.lib.format.open_memmap(
            args.output_dir / "sid_codeword_concat.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, dense_dim),
        )
        recon = np.lib.format.open_memmap(
            args.output_dir / "sid_reconstruction.npy",
            mode="w+",
            dtype=np.float32,
            shape=(n, args.code_dim),
        )
        for start, end in iter_ranges(n, args.chunk_size):
            chunk_codes = np.asarray(codes[start:end], dtype=np.int32)
            parts = []
            reconstruction = np.zeros((end - start, args.code_dim), dtype=np.float32)
            for level, centers in enumerate(rq_codebooks):
                selected = centers[chunk_codes[:, level]]
                parts.append(selected)
                reconstruction += selected
            selected_rotated = np.empty((end - start, args.code_dim), dtype=np.float32)
            for subspace in range(args.opq_subspaces):
                s0, s1 = subspace * subdim, (subspace + 1) * subdim
                selected = opq_codebooks[subspace][chunk_codes[:, args.rq_levels + subspace]]
                parts.append(selected)
                selected_rotated[:, s0:s1] = selected
            reconstruction += selected_rotated @ rotation.T
            dense[start:end] = np.concatenate(parts, axis=1).astype(np.float32)
            recon[start:end] = reconstruction
        dense.flush()
        recon.flush()

    write_item_to_sid_head(
        args.output_dir / "item_to_sid_head.csv",
        itemid,
        codes,
        offset_codes,
    )
    if args.write_full_csv:
        write_item_to_sid_full(
            args.output_dir / "item_to_sid.csv",
            itemid,
            codes,
            offset_codes,
            args.chunk_size,
        )

    collision_summary = (
        compute_collision_summary(np.asarray(codes)) if args.compute_collisions else None
    )
    offset_vocab_size = args.rq_levels * args.rq_clusters + args.opq_subspaces * args.opq_clusters
    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "data": {
            "num_items": int(n),
            "num_items_total": int(n_total),
            "title_shape": list(title.shape),
            "image_shape": list(image.shape),
            "item_feat_keys": list(feat.files),
        },
        "fusion": {
            "discrete_embedding_mode": args.discrete_embedding_mode,
            "fused_dim": int(fusion.fused_dim),
            "cat_emb_dim": args.cat_emb_dim,
            "label_emb_dim": args.label_emb_dim,
            "active_flag_cols": active_flag_cols,
            "weights": weights,
            "cat_vocab_sizes": cat_vocab_sizes.astype(int).tolist(),
            "label_vocab_size": int(label_vocab_size),
            "discrete_table_stats": discrete_table_stats,
        },
        "pca": {
            "code_dim": args.code_dim,
            "explained_variance_ratio_sum": float(ipca.explained_variance_ratio_.sum()),
            "top_explained_variance_ratio": ipca.explained_variance_ratio_[:10]
            .astype(float)
            .tolist(),
        },
        "rq_kmeans": {"per_level": rq_metrics},
        "opq": {
            "history": opq_metrics_history,
            "final_rotated_mse": opq_metrics_history[-1]["rotated_mse"],
            "final_original_space_mse": opq_metrics_history[-1]["original_space_mse"],
            "rotation_orthogonality_error": float(
                np.linalg.norm(rotation.T @ rotation - np.eye(args.code_dim, dtype=np.float32))
            ),
        },
        "sid": {
            "codes_shape": list(codes.shape),
            "rq_codebook_shape": [args.rq_levels, args.rq_clusters, args.code_dim],
            "opq_codebook_shape": [args.opq_subspaces, args.opq_clusters, subdim],
            "offset_vocab_size": int(offset_vocab_size),
            "categorical_tokens_per_item": int(args.rq_levels + args.opq_subspaces),
            "dense_concat_dim_if_saved": int(args.rq_levels * args.code_dim + args.opq_subspaces * subdim),
            "dense_reconstruction_dim_if_saved": int(args.code_dim),
        },
        "collisions": collision_summary,
        "elapsed_seconds": float(time.time() - start_time),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log("Done")
    log(f"  sid_codes.npy: {codes.shape}")
    log(f"  output_dir: {args.output_dir}")
    log(f"  elapsed_seconds: {metrics['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
