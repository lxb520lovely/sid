from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import iter_ranges, l2_normalize, resize_dim


@dataclass
class CatInputs:
    cat_ids: np.ndarray
    cat_vocab_sizes: np.ndarray
    cat_field_indices: list[int]


class FusionBuilder:
    def __init__(
        self,
        title: np.ndarray,
        image: np.ndarray,
        cat_ids: np.ndarray,
        cat_tables: list[np.ndarray],
        weights: dict[str, float],
    ) -> None:
        self.title = title
        self.image = image
        self.cat_ids = cat_ids
        self.cat_tables = cat_tables
        self.weights = weights

    @property
    def cat_feature_dim(self) -> int:
        return sum(table.shape[1] for table in self.cat_tables)

    @property
    def fused_dim(self) -> int:
        return self.title.shape[1] + self.image.shape[1] + self.cat_feature_dim

    def chunk(self, start: int, end: int) -> np.ndarray:
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
    return n


def prepare_cat_inputs(
    feat: np.lib.npyio.NpzFile,
    n: int,
    n_total: int,
    cat_fields: int,
) -> CatInputs:
    if cat_fields <= 0:
        raise ValueError("--cat-fields must be positive")

    cat_ids_all = np.asarray(feat["cat_ids"], dtype=np.int64)
    if cat_ids_all.ndim != 2:
        raise ValueError("item_feat cat_ids must be 2-D")
    if cat_ids_all.shape[0] != n_total:
        raise ValueError("item_feat cat_ids row count does not match embeddings")
    if cat_fields > cat_ids_all.shape[1]:
        raise ValueError(
            f"--cat-fields={cat_fields} exceeds cat_ids columns "
            f"({cat_ids_all.shape[1]})"
        )

    raw_cat_vocab_sizes = np.asarray(feat["cat_vocab_sizes"], dtype=np.int64)
    if cat_fields > raw_cat_vocab_sizes.shape[0]:
        raise ValueError(
            f"--cat-fields={cat_fields} exceeds cat_vocab_sizes length "
            f"({raw_cat_vocab_sizes.shape[0]})"
        )

    return CatInputs(
        cat_ids=np.ascontiguousarray(cat_ids_all[:n, :cat_fields]),
        cat_vocab_sizes=raw_cat_vocab_sizes[:cat_fields],
        cat_field_indices=list(range(cat_fields)),
    )


def make_random_table(vocab_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    table = rng.normal(0.0, 1.0 / math.sqrt(max(dim, 1)), size=(vocab_size, dim))
    table[0] *= 0.5
    return table.astype(np.float32)


def semantic_base_chunk(title: np.ndarray, image: np.ndarray, start: int, end: int) -> np.ndarray:
    """Base semantic vector used to aggregate discrete cat embeddings."""
    title_norm = l2_normalize(np.asarray(title[start:end], dtype=np.float32))
    image_norm = l2_normalize(np.asarray(image[start:end], dtype=np.float32))
    return ((title_norm + image_norm) * 0.5).astype(np.float32)


def build_random_tables(
    cat_vocab_sizes: np.ndarray,
    cat_emb_dim: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cat_tables = [
        make_random_table(int(vocab), cat_emb_dim, rng)
        for vocab in cat_vocab_sizes.tolist()
    ]
    return cat_tables, {"mode": "random"}


def build_semantic_mean_tables(
    title: np.ndarray,
    image: np.ndarray,
    cat_ids: np.ndarray,
    n: int,
    cat_vocab_sizes: np.ndarray,
    cat_emb_dim: int,
    chunk_size: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Aggregate title/image semantic base vectors into cat tables."""
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

    return cat_tables, {
        "mode": "semantic_mean",
        "semantic_base_dim": int(base_dim),
        "cat_stats": cat_stats,
    }


def build_cat_tables(
    mode: str,
    title: np.ndarray,
    image: np.ndarray,
    cat_ids: np.ndarray,
    n: int,
    cat_vocab_sizes: np.ndarray,
    cat_emb_dim: int,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if mode == "random":
        return build_random_tables(cat_vocab_sizes, cat_emb_dim, rng)
    if mode == "semantic_mean":
        return build_semantic_mean_tables(
            title=title,
            image=image,
            cat_ids=cat_ids,
            n=n,
            cat_vocab_sizes=cat_vocab_sizes,
            cat_emb_dim=cat_emb_dim,
            chunk_size=chunk_size,
        )
    raise ValueError(f"Unsupported discrete embedding mode: {mode}")
