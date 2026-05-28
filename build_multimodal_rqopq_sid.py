#!/usr/bin/env python3
"""Streaming multi-modal RQ-OPQ SID builder.

Baseline 4 input:
  title_emb.npy  -> dense title embedding, shape (N, 256)
  image_emb.npy  -> dense image embedding, shape (N, 256)
  item_feat.npz  -> itemid, cat_ids
  itemid.npy     -> raw item-id mapping for each row

The script builds a continuous multi-modal item representation with:
  L2(title) + L2(image) + concatenated per-field cat_id embeddings

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
DEFAULT_TITLE_WEIGHT = 1.0
DEFAULT_IMAGE_WEIGHT = 1.0
DEFAULT_CAT_WEIGHT = 0.20


# CLI options for inputs, feature fusion, quantization, and artifact writing.
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

    parser.add_argument(
        "--cat-fields",
        type=int,
        default=4,
        help="Use the first N columns from item_feat.npz:cat_ids.",
    )
    parser.add_argument(
        "--cat-emb-dim",
        type=int,
        default=16,
        help="Embedding dimension per cat_id field.",
    )
    parser.add_argument(
        "--discrete-embedding-mode",
        choices=("semantic_mean", "random"),
        default="random",
        help="How to map cat_ids to dense vectors.",
    )
    parser.add_argument("--title-weight", type=float, default=DEFAULT_TITLE_WEIGHT)
    parser.add_argument("--image-weight", type=float, default=DEFAULT_IMAGE_WEIGHT)
    parser.add_argument("--cat-weight", type=float, default=DEFAULT_CAT_WEIGHT)

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


# Lightweight helpers used by all streaming stages.
def log(msg: str) -> None:
    """Print progress immediately so long streaming jobs expose live status."""
    print(msg, flush=True)


def iter_ranges(n: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield half-open row ranges used to scan large arrays in fixed-size chunks."""
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normalize each row to unit length while avoiding division by zero."""
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def entropy_from_counts(counts: np.ndarray) -> float:
    """Compute Shannon entropy of non-empty cluster counts for usage balance metrics."""
    positive = counts[counts > 0].astype(np.float64)
    if positive.size == 0:
        return 0.0
    p = positive / positive.sum()
    return float(-(p * np.log2(p)).sum())


def make_random_table(vocab_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Create reproducible random embeddings for categorical ids.

    This is a simple non-semantic baseline for mapping discrete cat_ids into
    dense vectors before feature fusion.
    """
    table = rng.normal(0.0, 1.0 / math.sqrt(max(dim, 1)), size=(vocab_size, dim))
    table[0] *= 0.5
    return table.astype(np.float32)


def resize_dim(x: np.ndarray, dim: int) -> np.ndarray:
    """Resize feature vectors by truncating extra dimensions or padding zeros.

    The operation is deterministic and keeps the first dimensions unchanged.
    """
    if x.shape[1] == dim:
        return x.astype(np.float32, copy=False)
    if x.shape[1] > dim:
        return x[:, :dim].astype(np.float32, copy=False)
    out = np.zeros((x.shape[0], dim), dtype=np.float32)
    out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
    return out


def semantic_base_chunk(title: np.ndarray, image: np.ndarray, start: int, end: int) -> np.ndarray:
    """Build semantic base vectors for one item chunk.

    Each row is the average of L2-normalized title and image embeddings. These
    vectors are later averaged by cat_id in semantic_mean mode.
    """
    title_norm = l2_normalize(np.asarray(title[start:end], dtype=np.float32))
    image_norm = l2_normalize(np.asarray(image[start:end], dtype=np.float32))
    return ((title_norm + image_norm) * 0.5).astype(np.float32)


def build_semantic_mean_tables(
    title: np.ndarray,
    image: np.ndarray,
    cat_ids: np.ndarray,
    n: int,
    cat_vocab_sizes: np.ndarray,
    cat_emb_dim: int,
    chunk_size: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Build semantic_mean embedding tables for categorical fields.

    For each categorical field and each cat_id, the table stores the mean
    title/image semantic vector of items carrying that id. Unseen ids fall back
    to the global semantic mean, then vectors are normalized and resized.
    """
    base_dim = title.shape[1]
    cat_sums = [
        np.zeros((int(vocab), base_dim), dtype=np.float64)
        for vocab in cat_vocab_sizes.tolist()
    ]
    cat_counts = [
        np.zeros(int(vocab), dtype=np.int64) for vocab in cat_vocab_sizes.tolist()
    ]
    global_sum = np.zeros(base_dim, dtype=np.float64)
    global_count = 0

    for start, end in iter_ranges(n, chunk_size):
        base = semantic_base_chunk(title, image, start, end)
        cat_chunk = np.asarray(cat_ids[start:end], dtype=np.int64)
        global_sum += base.sum(axis=0, dtype=np.float64)
        global_count += end - start
        for field_idx in range(cat_chunk.shape[1]):
            np.add.at(cat_sums[field_idx], cat_chunk[:, field_idx], base)
            cat_counts[field_idx] += np.bincount(
                cat_chunk[:, field_idx], minlength=cat_counts[field_idx].shape[0]
            )
    
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

    stats = {
        "semantic_base_dim": int(base_dim),
        "cat_stats": cat_stats,
    }
    return cat_tables, stats


def validate_inputs(
    title: np.ndarray,
    image: np.ndarray,
    itemid: np.ndarray,
    feat: np.lib.npyio.NpzFile,
) -> int:
    """Validate that title, image, itemid, and item_feat rows describe the same items.

    The pipeline assumes row i across all inputs belongs to the same item, so a
    mismatch would corrupt the generated SID assignments.
    """
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
    return n


class FusionBuilder:
    """Construct fused multi-modal feature chunks on demand.

    The class keeps references to the source arrays and categorical embedding
    tables, then produces weighted concatenated features for streaming PCA.
    """

    def __init__(
        self,
        title: np.ndarray,
        image: np.ndarray,
        cat_ids: np.ndarray,
        cat_tables: list[np.ndarray],
        weights: dict[str, float],
    ) -> None:
        """Store source arrays, categorical tables, and modality weights."""
        self.title = title
        self.image = image
        self.cat_ids = cat_ids
        self.cat_tables = cat_tables
        self.weights = weights

    @property
    def cat_feature_dim(self) -> int:
        """Total dense dimension contributed by all categorical fields."""
        return sum(table.shape[1] for table in self.cat_tables)

    @property
    def fused_dim(self) -> int:
        """Total dimension before PCA: title + image + categorical embeddings."""
        return (
            self.title.shape[1]
            + self.image.shape[1]
            + self.cat_feature_dim
        )

    def chunk(self, start: int, end: int) -> np.ndarray:
        """Build one weighted fused feature chunk without materializing all rows.

        Title and image vectors are L2-normalized per row. Categorical ids are
        looked up in their per-field tables and concatenated, then all modalities
        are scaled by their configured weights.
        """
        title = l2_normalize(np.asarray(self.title[start:end], dtype=np.float32))
        image = l2_normalize(np.asarray(self.image[start:end], dtype=np.float32))
        cat_ids = np.asarray(self.cat_ids[start:end], dtype=np.int64)

        cat_emb = np.concatenate(
            [
                table[cat_ids[:, field_idx]]
                for field_idx, table in enumerate(self.cat_tables)
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        return np.concatenate(
            [
                title * self.weights["title"],
                image * self.weights["image"],
                cat_emb * self.weights["cat"],
            ],
            axis=1,
        ).astype(np.float32)


def nearest_centers(x: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each row to the nearest codebook center.

    Returns both the integer labels and the corresponding squared L2 distances.
    This is used after KMeans training to write SID codes and compute MSE.
    """
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
    """Compute the current RQ residual for a projected item chunk.

    Starting from the PCA-projected vector, subtract each previously selected RQ
    codeword so the next RQ level learns what earlier levels did not explain.
    """
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
    """Fit IncrementalPCA over streamed fused features.

    This reduces the high-dimensional multi-modal representation to code_dim
    before quantization without loading the full fused matrix into memory.
    """
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
    """Transform all fused features through PCA and save the projected memmap.

    The returned array is backed by projected.npy and reused by RQ/OPQ stages.
    """
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
    """Train one MiniBatchKMeans codebook from a streaming data iterator.

    The iterator is recreated each epoch so the codebook can make multiple
    passes over residuals or subspace vectors.
    """
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
    """Summarize how well a codebook is used.

    The metrics include used code count, utilization ratio, entropy, and min/max
    assigned cluster sizes.
    """
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
    """Convert local per-position codes into a single global token vocabulary.

    RQ and OPQ positions can reuse local ids like 0 or 42, so each position gets
    an offset to make its token ids distinct for downstream sequence models.
    """
    offsets = []
    for level in range(codes.shape[1]):
        if level < rq_levels:
            offsets.append(level * rq_clusters)
        else:
            offsets.append(rq_levels * rq_clusters + (level - rq_levels) * opq_clusters)
    return (codes + np.asarray(offsets, dtype=np.int32)[None, :]).astype(np.int32)


def compute_collision_summary(codes: np.ndarray) -> dict[str, Any]:
    """Compute exact collisions among full SID sequences.

    A collision means two or more items share the same complete SID tuple. This
    uses np.unique over all rows and can be memory-heavy on full datasets.
    """
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
    """Write a small human-readable item-to-SID preview CSV.

    The preview contains raw local SID tokens, offset global tokens, and joined
    string forms for the first n_rows items.
    """
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
    """Write the full item-to-SID CSV in chunks.

    This mirrors the preview format but emits every item, so it is optional for
    large datasets.
    """
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
    """Run the full SID build pipeline from raw multi-modal inputs to artifacts."""
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # OPQ splits the PCA space evenly across subspaces.
    if args.code_dim % args.opq_subspaces != 0:
        raise ValueError("--code-dim must be divisible by --opq-subspaces")

    # Load large arrays with mmap so processing stays chunk-based.
    title = np.load(args.title_emb, mmap_mode="r", allow_pickle=False)
    image = np.load(args.image_emb, mmap_mode="r", allow_pickle=False)
    itemid = np.load(args.itemid, mmap_mode="r", allow_pickle=False)
    feat = np.load(args.item_feat, allow_pickle=False)
    n_total = validate_inputs(title, image, itemid, feat)
    n = min(n_total, args.max_items) if args.max_items else n_total

    log(f"Rows: using {n:,} / {n_total:,}")
    log(f"Title shape: {title.shape}, image shape: {image.shape}")

    # Select and validate the categorical fields used in fusion.
    if args.cat_fields <= 0:
        raise ValueError("--cat-fields must be positive")
    cat_ids_all = np.asarray(feat["cat_ids"], dtype=np.int64)
    if cat_ids_all.ndim != 2:
        raise ValueError("item_feat cat_ids must be 2-D")
    if cat_ids_all.shape[0] != n_total:
        raise ValueError("item_feat cat_ids row count does not match embeddings")
    if args.cat_fields > cat_ids_all.shape[1]:
        raise ValueError(
            f"--cat-fields={args.cat_fields} exceeds cat_ids columns "
            f"({cat_ids_all.shape[1]})"
        )

    raw_cat_vocab_sizes = np.asarray(feat["cat_vocab_sizes"], dtype=np.int64)
    if args.cat_fields > raw_cat_vocab_sizes.shape[0]:
        raise ValueError(
            f"--cat-fields={args.cat_fields} exceeds cat_vocab_sizes length "
            f"({raw_cat_vocab_sizes.shape[0]})"
        )
    cat_vocab_sizes = raw_cat_vocab_sizes[: args.cat_fields]
    cat_field_indices = list(range(args.cat_fields))
    cat_ids = np.ascontiguousarray(cat_ids_all[:n, : args.cat_fields])
    del cat_ids_all
    log(f"Using cat_id columns: {cat_field_indices}")

    rng = np.random.default_rng(args.seed)
    discrete_table_stats: dict[str, Any]
    if args.discrete_embedding_mode == "random":
        # Random mode is a reproducible categorical baseline.
        log("Building random cat embedding tables")
        cat_tables = [
            make_random_table(int(vocab), args.cat_emb_dim, rng)
            for vocab in cat_vocab_sizes.tolist()
        ]
        discrete_table_stats = {"mode": "random"}
    else:
        # Semantic mode averages title/image vectors for each categorical id.
        log("Building semantic_mean cat embedding tables")
        cat_tables, discrete_table_stats = build_semantic_mean_tables(
            title=title,
            image=image,
            cat_ids=cat_ids,
            n=n,
            cat_vocab_sizes=cat_vocab_sizes,
            cat_emb_dim=args.cat_emb_dim,
            chunk_size=args.chunk_size,
        )
        discrete_table_stats["mode"] = "semantic_mean"

    weights = {
        "title": args.title_weight,
        "image": args.image_weight,
        "cat": args.cat_weight,
    }
    log(
        "Modality weights: "
        f"title={weights['title']:.8g}, "
        f"image={weights['image']:.8g}, "
        f"cat={weights['cat']:.8g}"
    )
    fusion = FusionBuilder(
        title=title,
        image=image,
        cat_ids=cat_ids,
        cat_tables=cat_tables,
        weights=weights,
    )
    log(f"Cat feature dim before PCA: {fusion.cat_feature_dim}")
    log(f"Fused feature dim before PCA: {fusion.fused_dim}")

    # Reduce fused multi-modal vectors before quantization.
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
        # RQ learns one codebook per residual level.
        log(f"Starting RQ level {level + 1}/{args.rq_levels}")

        def rq_train_iter(level=level):
            """Stream residual vectors for the current RQ level."""
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
            # Assign current residuals to this level's nearest centers.
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
        # OPQ alternates between subspace codebooks and rotation updates.
        log(f"Starting OPQ outer iteration {outer + 1}/{args.opq_outer_iter}")
        subspace_metrics = []
        for subspace in range(args.opq_subspaces):
            s0, s1 = subspace * subdim, (subspace + 1) * subdim

            def opq_train_iter(subspace=subspace, s0=s0, s1=s1):
                """Stream one rotated residual subspace for OPQ codebook training."""
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
            # Quantize the RQ residual inside each rotated subspace.
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
            # Orthogonal Procrustes update for the next OPQ iteration.
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = (u @ vt).astype(np.float32)

    codes.flush()
    # Offset codes give each SID position its own token-id range.
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

    # Persist codebooks, projection, fusion metadata, and item ids.
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
        cat_field_indices=np.asarray(cat_field_indices, dtype=np.int32),
        cat_vocab_sizes=cat_vocab_sizes.astype(np.int64),
        cat_feature_dim=np.asarray([fusion.cat_feature_dim], dtype=np.int32),
        discrete_embedding_mode=np.asarray([args.discrete_embedding_mode]),
        weight_names=np.asarray(["title", "image", "cat"]),
        weights=np.asarray(
            [weights["title"], weights["image"], weights["cat"]],
            dtype=np.float32,
        ),
    )

    if args.save_dense:
        # Dense reconstruction artifacts are large and therefore optional.
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
        # Full CSV is useful for inspection but expensive at full scale.
        write_item_to_sid_full(
            args.output_dir / "item_to_sid.csv",
            itemid,
            codes,
            offset_codes,
            args.chunk_size,
        )

    # Exact collision counting materializes all SID rows, so it is optional.
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
            "cat_fields": int(args.cat_fields),
            "cat_field_indices": cat_field_indices,
            "cat_emb_dim": args.cat_emb_dim,
            "cat_feature_dim": int(fusion.cat_feature_dim),
            "weights": weights,
            "cat_vocab_sizes": cat_vocab_sizes.astype(int).tolist(),
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
