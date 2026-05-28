from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import iter_ranges


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


def save_projection(
    path: Path,
    mean: np.ndarray,
    components: np.ndarray,
    explained_variance: np.ndarray,
    explained_variance_ratio: np.ndarray,
    input_dim: int,
) -> None:
    np.savez_compressed(
        path,
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=explained_variance.astype(np.float32),
        explained_variance_ratio=explained_variance_ratio.astype(np.float32),
        input_dim=np.asarray([input_dim], dtype=np.int32),
    )


def save_fusion_tables(
    path: Path,
    cat_tables: list[np.ndarray],
    cat_field_indices: list[int],
    cat_vocab_sizes: np.ndarray,
    cat_feature_dim: int,
    discrete_embedding_mode: str,
    weights: dict[str, float],
) -> None:
    np.savez_compressed(
        path,
        **{f"cat_table_{i}": table for i, table in enumerate(cat_tables)},
        cat_field_indices=np.asarray(cat_field_indices, dtype=np.int32),
        cat_vocab_sizes=cat_vocab_sizes.astype(np.int64),
        cat_feature_dim=np.asarray([cat_feature_dim], dtype=np.int32),
        discrete_embedding_mode=np.asarray([discrete_embedding_mode]),
        weight_names=np.asarray(["title", "image", "cat"]),
        weights=np.asarray(
            [weights["title"], weights["image"], weights["cat"]],
            dtype=np.float32,
        ),
    )


def save_dense_artifacts(
    output_dir: Path,
    codes: np.ndarray,
    rq_codebooks: list[np.ndarray],
    opq_codebooks: np.ndarray,
    rotation: np.ndarray,
    rq_levels: int,
    code_dim: int,
    opq_subspaces: int,
    chunk_size: int,
) -> None:
    subdim = code_dim // opq_subspaces
    dense_dim = rq_levels * code_dim + opq_subspaces * subdim
    dense = np.lib.format.open_memmap(
        output_dir / "sid_codeword_concat.npy",
        mode="w+",
        dtype=np.float32,
        shape=(codes.shape[0], dense_dim),
    )
    recon = np.lib.format.open_memmap(
        output_dir / "sid_reconstruction.npy",
        mode="w+",
        dtype=np.float32,
        shape=(codes.shape[0], code_dim),
    )
    for start, end in iter_ranges(codes.shape[0], chunk_size):
        chunk_codes = np.asarray(codes[start:end], dtype=np.int32)
        parts = []
        reconstruction = np.zeros((end - start, code_dim), dtype=np.float32)
        for level, centers in enumerate(rq_codebooks):
            selected = centers[chunk_codes[:, level]]
            parts.append(selected)
            reconstruction += selected
        selected_rotated = np.empty((end - start, code_dim), dtype=np.float32)
        for subspace in range(opq_subspaces):
            s0, s1 = subspace * subdim, (subspace + 1) * subdim
            selected = opq_codebooks[subspace][chunk_codes[:, rq_levels + subspace]]
            parts.append(selected)
            selected_rotated[:, s0:s1] = selected
        reconstruction += selected_rotated @ rotation.T
        dense[start:end] = np.concatenate(parts, axis=1).astype(np.float32)
        recon[start:end] = reconstruction
    dense.flush()
    recon.flush()
