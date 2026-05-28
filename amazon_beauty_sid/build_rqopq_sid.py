from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import IncrementalPCA

from multimodal_rqopq_sid_code.common import iter_ranges
from multimodal_rqopq_sid_code.quantization import (
    compute_collision_summary,
    train_opq,
    train_rq,
)


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RQ-OPQ semantic IDs from item_embeddings.npy."
    )
    parser.add_argument(
        "--item-embeddings",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed/item_embeddings.npy"),
    )
    parser.add_argument(
        "--itemid",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed/itemid.npy"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/sid_rqopq"),
    )
    parser.add_argument("--code-dim", type=int, default=32)
    parser.add_argument("--rq-clusters", type=int, default=256)
    parser.add_argument("--rq-levels", type=int, default=3)
    parser.add_argument("--opq-subspaces", type=int, default=2)
    parser.add_argument("--opq-clusters", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--pca-epochs", type=int, default=1)
    parser.add_argument("--rq-epochs", type=int, default=3)
    parser.add_argument("--opq-outer-iter", type=int, default=3)
    parser.add_argument("--opq-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--append-collision-suffix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append a final per-collision suffix token. Disabled by default so "
            "the SID stays equal to rq_levels + opq_subspaces."
        ),
    )
    parser.add_argument("--write-full-csv", action="store_true")
    return parser.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)


def embedding_chunk(embeddings: np.ndarray, start: int, end: int, normalize: bool) -> np.ndarray:
    chunk = np.asarray(embeddings[start:end], dtype=np.float32)
    if normalize:
        chunk = l2_normalize(chunk)
    return chunk


def project_embeddings(
    embeddings: np.ndarray,
    output_dir: Path,
    code_dim: int,
    chunk_size: int,
    epochs: int,
    normalize: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    n, input_dim = embeddings.shape
    if code_dim <= 0:
        raise ValueError("--code-dim must be positive")
    if code_dim > min(input_dim, n - 1):
        raise ValueError(
            f"--code-dim={code_dim} exceeds min(input_dim, n-1)="
            f"{min(input_dim, n - 1)}"
        )

    ipca = IncrementalPCA(n_components=code_dim)
    for epoch in range(epochs):
        print(f"Fitting item PCA epoch {epoch + 1}/{epochs}", flush=True)
        for start, end in iter_ranges(n, chunk_size):
            ipca.partial_fit(embedding_chunk(embeddings, start, end, normalize))

    projected = np.lib.format.open_memmap(
        output_dir / "projected.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, code_dim),
    )
    for start, end in iter_ranges(n, chunk_size):
        projected[start:end] = ipca.transform(
            embedding_chunk(embeddings, start, end, normalize)
        ).astype(np.float32)
    projected.flush()

    np.savez_compressed(
        output_dir / "projection.npz",
        mean=ipca.mean_.astype(np.float32),
        components=ipca.components_.astype(np.float32),
        explained_variance=ipca.explained_variance_.astype(np.float32),
        explained_variance_ratio=ipca.explained_variance_ratio_.astype(np.float32),
        input_dim=np.asarray([input_dim], dtype=np.int32),
        normalize=np.asarray([normalize]),
    )
    return projected, {
        "input_dim": int(input_dim),
        "code_dim": int(code_dim),
        "explained_variance_ratio_sum": float(ipca.explained_variance_ratio_.sum()),
        "top_explained_variance_ratio": ipca.explained_variance_ratio_[:10]
        .astype(float)
        .tolist(),
    }


def offset_codes_by_sizes(codes: np.ndarray, codebook_sizes: list[int]) -> np.ndarray:
    if codes.shape[1] != len(codebook_sizes):
        raise ValueError("codes width and codebook_sizes length differ")
    offsets = np.asarray([0] + list(np.cumsum(codebook_sizes)[:-1]), dtype=np.int32)
    return (np.asarray(codes, dtype=np.int32) + offsets[None, :]).astype(np.int32)


def write_item_to_sid(
    path: Path,
    item_ids: np.ndarray,
    codes: np.ndarray,
    offset_codes: np.ndarray,
    full: bool,
) -> None:
    n = codes.shape[0] if full else min(10000, codes.shape[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["row_id", "itemid"]
        header += [f"sid_{i}" for i in range(codes.shape[1])]
        header += [f"sid_offset_{i}" for i in range(codes.shape[1])]
        header += ["sid", "sid_offset"]
        writer.writerow(header)
        for row_id in range(n):
            sid = codes[row_id].astype(int).tolist()
            sid_offset = offset_codes[row_id].astype(int).tolist()
            writer.writerow(
                [row_id, int(item_ids[row_id])]
                + sid
                + sid_offset
                + ["-".join(map(str, sid)), "-".join(map(str, sid_offset))]
            )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.opq_subspaces < 0:
        raise ValueError("--opq-subspaces must be >= 0")
    if args.opq_subspaces and args.code_dim % args.opq_subspaces != 0:
        raise ValueError("--code-dim must be divisible by --opq-subspaces")

    embeddings = np.load(args.item_embeddings, mmap_mode="r", allow_pickle=False)
    item_ids = np.load(args.itemid, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError("item embeddings must be 2-D")
    if item_ids.shape[0] != embeddings.shape[0]:
        raise ValueError("itemid row count does not match item embeddings")

    n = embeddings.shape[0]
    print(f"Rows: {n:,}, embedding shape: {embeddings.shape}", flush=True)
    projected, pca_metrics = project_embeddings(
        embeddings=embeddings,
        output_dir=args.output_dir,
        code_dim=args.code_dim,
        chunk_size=args.chunk_size,
        epochs=args.pca_epochs,
        normalize=not args.no_normalize,
    )

    raw_width = args.rq_levels + args.opq_subspaces
    raw_codes = np.lib.format.open_memmap(
        args.output_dir / "sid_codes_quantized.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n, raw_width),
    )
    rq = train_rq(
        projected=projected,
        codes=raw_codes,
        n=n,
        rq_levels=args.rq_levels,
        rq_clusters=args.rq_clusters,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        epochs=args.rq_epochs,
        seed=args.seed,
    )

    opq_metrics = []
    opq_codebooks = None
    opq_rotation = np.eye(args.code_dim, dtype=np.float32)
    if args.opq_subspaces:
        opq = train_opq(
            projected=projected,
            codes=raw_codes,
            rq_codebooks=rq.codebooks,
            n=n,
            code_dim=args.code_dim,
            rq_levels=args.rq_levels,
            opq_subspaces=args.opq_subspaces,
            opq_clusters=args.opq_clusters,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            outer_iter=args.opq_outer_iter,
            epochs=args.opq_epochs,
            seed=args.seed,
        )
        opq_metrics = opq.metrics_history
        opq_codebooks = opq.codebooks
        opq_rotation = opq.rotation
    raw_codes.flush()

    raw_collision_summary = compute_collision_summary(np.asarray(raw_codes))
    codebook_sizes = [args.rq_clusters] * args.rq_levels + [
        args.opq_clusters
    ] * args.opq_subspaces
    if args.append_collision_suffix:
        from collections import defaultdict

        groups: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
        for row_idx, row in enumerate(np.asarray(raw_codes)):
            groups[tuple(int(x) for x in row)].append(row_idx)
        suffix_size = max((len(rows) for rows in groups.values()), default=1)
        suffix = np.zeros((raw_codes.shape[0], 1), dtype=np.int32)
        for rows in groups.values():
            for suffix_idx, row_idx in enumerate(rows):
                suffix[row_idx, 0] = suffix_idx
        final_codes = np.concatenate([np.asarray(raw_codes, dtype=np.int32), suffix], axis=1)
        codebook_sizes.append(int(suffix_size))
    else:
        final_codes = np.asarray(raw_codes, dtype=np.int32)
        suffix_size = 0

    final_collision_summary = compute_collision_summary(final_codes)
    np.save(args.output_dir / "sid_codes.npy", final_codes.astype(np.int32))
    offset_codes = offset_codes_by_sizes(final_codes, codebook_sizes)
    np.save(args.output_dir / "sid_codes_offset.npy", offset_codes.astype(np.int32))
    np.save(args.output_dir / "sid_codebook_sizes.npy", np.asarray(codebook_sizes, dtype=np.int32))
    np.save(args.output_dir / "itemid.npy", np.asarray(item_ids, dtype=np.int64))
    np.save(args.output_dir / "rq_codebooks.npy", np.stack(rq.codebooks).astype(np.float32))
    if opq_codebooks is not None:
        np.save(args.output_dir / "opq_codebooks.npy", opq_codebooks.astype(np.float32))
    np.save(args.output_dir / "opq_rotation.npy", opq_rotation.astype(np.float32))

    write_item_to_sid(
        args.output_dir / "item_to_sid_head.csv",
        item_ids=item_ids,
        codes=final_codes,
        offset_codes=offset_codes,
        full=False,
    )
    if args.write_full_csv:
        write_item_to_sid(
            args.output_dir / "item_to_sid.csv",
            item_ids=item_ids,
            codes=final_codes,
            offset_codes=offset_codes,
            full=True,
        )

    metrics = {
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "data": {
            "item_embeddings": str(args.item_embeddings),
            "itemid": str(args.itemid),
            "num_items": int(n),
            "embedding_shape": list(embeddings.shape),
        },
        "pca": pca_metrics,
        "rq": {"per_level": rq.metrics},
        "opq": {
            "enabled": bool(args.opq_subspaces),
            "history": opq_metrics,
        },
        "sid": {
            "raw_codes_shape": list(raw_codes.shape),
            "final_codes_shape": list(final_codes.shape),
            "codebook_sizes": codebook_sizes,
            "offset_vocab_size": int(sum(codebook_sizes)),
            "append_collision_suffix": bool(args.append_collision_suffix),
            "collision_suffix_size": int(suffix_size),
            "raw_collisions": raw_collision_summary,
            "final_collisions": final_collision_summary,
        },
        "elapsed_seconds": float(time.time() - start_time),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metrics["sid"], indent=2), flush=True)
    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
