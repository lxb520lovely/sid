from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from .common import log
from .features import (
    FusionBuilder,
    build_cat_tables,
    prepare_cat_inputs,
    validate_inputs,
)
from .outputs import (
    save_dense_artifacts,
    save_fusion_tables,
    save_projection,
    write_item_to_sid_full,
    write_item_to_sid_head,
)
from .quantization import (
    compute_collision_summary,
    fit_incremental_pca,
    train_opq,
    train_rq,
    transform_projected,
    write_offset_codes,
)


def load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, int, int]:
    title = np.load(args.title_emb, mmap_mode="r", allow_pickle=False)
    image = np.load(args.image_emb, mmap_mode="r", allow_pickle=False)
    itemid = np.load(args.itemid, mmap_mode="r", allow_pickle=False)
    feat = np.load(args.item_feat, allow_pickle=False)
    n_total = validate_inputs(title, image, itemid, feat)
    n = min(n_total, args.max_items) if args.max_items else n_total
    return title, image, itemid, feat, n_total, n


def build_fusion(
    args: argparse.Namespace,
    title: np.ndarray,
    image: np.ndarray,
    feat: Any,
    n_total: int,
    n: int,
) -> tuple[FusionBuilder, list[np.ndarray], dict[str, Any], list[int], np.ndarray, dict[str, float]]:
    cat_inputs = prepare_cat_inputs(
        feat=feat,
        n=n,
        n_total=n_total,
        cat_fields=args.cat_fields,
    )
    log(f"Using cat_id columns: {cat_inputs.cat_field_indices}")

    rng = np.random.default_rng(args.seed)
    if args.discrete_embedding_mode == "random":
        log("Building random cat embedding tables")
    else:
        log("Building semantic_mean cat embedding tables")
    cat_tables, discrete_table_stats = build_cat_tables(
        mode=args.discrete_embedding_mode,
        title=title,
        image=image,
        cat_ids=cat_inputs.cat_ids,
        n=n,
        cat_vocab_sizes=cat_inputs.cat_vocab_sizes,
        cat_emb_dim=args.cat_emb_dim,
        chunk_size=args.chunk_size,
        rng=rng,
    )

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
        cat_ids=cat_inputs.cat_ids,
        cat_tables=cat_tables,
        weights=weights,
    )
    log(f"Cat feature dim before PCA: {fusion.cat_feature_dim}")
    log(f"Fused feature dim before PCA: {fusion.fused_dim}")
    return (
        fusion,
        cat_tables,
        discrete_table_stats,
        cat_inputs.cat_field_indices,
        cat_inputs.cat_vocab_sizes,
        weights,
    )


def write_artifacts(
    args: argparse.Namespace,
    itemid: np.ndarray,
    fusion: FusionBuilder,
    cat_tables: list[np.ndarray],
    cat_field_indices: list[int],
    cat_vocab_sizes: np.ndarray,
    weights: dict[str, float],
    ipca: Any,
    codes: np.ndarray,
    offset_codes: np.ndarray,
    rq_codebooks: list[np.ndarray],
    opq_codebooks: np.ndarray,
    rotation: np.ndarray,
) -> None:
    np.save(args.output_dir / "rq_codebooks.npy", np.stack(rq_codebooks).astype(np.float32))
    np.save(args.output_dir / "opq_codebooks.npy", opq_codebooks.astype(np.float32))
    np.save(args.output_dir / "opq_rotation.npy", rotation.astype(np.float32))
    np.save(args.output_dir / "itemid.npy", np.asarray(itemid[: codes.shape[0]], dtype=np.int64))
    save_projection(
        args.output_dir / "projection.npz",
        mean=ipca.mean_,
        components=ipca.components_,
        explained_variance=ipca.explained_variance_,
        explained_variance_ratio=ipca.explained_variance_ratio_,
        input_dim=fusion.fused_dim,
    )
    save_fusion_tables(
        args.output_dir / "fusion_tables.npz",
        cat_tables=cat_tables,
        cat_field_indices=cat_field_indices,
        cat_vocab_sizes=cat_vocab_sizes,
        cat_feature_dim=fusion.cat_feature_dim,
        discrete_embedding_mode=args.discrete_embedding_mode,
        weights=weights,
    )

    if args.save_dense:
        save_dense_artifacts(
            output_dir=args.output_dir,
            codes=codes,
            rq_codebooks=rq_codebooks,
            opq_codebooks=opq_codebooks,
            rotation=rotation,
            rq_levels=args.rq_levels,
            code_dim=args.code_dim,
            opq_subspaces=args.opq_subspaces,
            chunk_size=args.chunk_size,
        )

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


def build_metrics(
    args: argparse.Namespace,
    title: np.ndarray,
    image: np.ndarray,
    feat: Any,
    n: int,
    n_total: int,
    fusion: FusionBuilder,
    cat_field_indices: list[int],
    cat_vocab_sizes: np.ndarray,
    weights: dict[str, float],
    discrete_table_stats: dict[str, Any],
    ipca: Any,
    rq_metrics: list[dict[str, Any]],
    opq_metrics_history: list[dict[str, Any]],
    rotation: np.ndarray,
    codes: np.ndarray,
    collision_summary: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    subdim = args.code_dim // args.opq_subspaces
    offset_vocab_size = args.rq_levels * args.rq_clusters + args.opq_subspaces * args.opq_clusters
    return {
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
        "elapsed_seconds": float(elapsed_seconds),
    }


def run(args: argparse.Namespace) -> None:
    start_time = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.code_dim % args.opq_subspaces != 0:
        raise ValueError("--code-dim must be divisible by --opq-subspaces")

    title, image, itemid, feat, n_total, n = load_inputs(args)
    log(f"Rows: using {n:,} / {n_total:,}")
    log(f"Title shape: {title.shape}, image shape: {image.shape}")

    fusion, cat_tables, discrete_table_stats, cat_field_indices, cat_vocab_sizes, weights = (
        build_fusion(args, title, image, feat, n_total, n)
    )

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
    rq = train_rq(
        projected=projected,
        codes=codes,
        n=n,
        rq_levels=args.rq_levels,
        rq_clusters=args.rq_clusters,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        epochs=args.rq_epochs,
        seed=args.seed,
    )
    opq = train_opq(
        projected=projected,
        codes=codes,
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
    codes.flush()

    offset_codes = write_offset_codes(
        args.output_dir / "sid_codes_offset.npy",
        codes=codes,
        rq_levels=args.rq_levels,
        rq_clusters=args.rq_clusters,
        opq_clusters=args.opq_clusters,
        chunk_size=args.chunk_size,
    )
    write_artifacts(
        args=args,
        itemid=itemid,
        fusion=fusion,
        cat_tables=cat_tables,
        cat_field_indices=cat_field_indices,
        cat_vocab_sizes=cat_vocab_sizes,
        weights=weights,
        ipca=ipca,
        codes=codes,
        offset_codes=offset_codes,
        rq_codebooks=rq.codebooks,
        opq_codebooks=opq.codebooks,
        rotation=opq.rotation,
    )

    collision_summary = (
        compute_collision_summary(np.asarray(codes)) if args.compute_collisions else None
    )
    metrics = build_metrics(
        args=args,
        title=title,
        image=image,
        feat=feat,
        n=n,
        n_total=n_total,
        fusion=fusion,
        cat_field_indices=cat_field_indices,
        cat_vocab_sizes=cat_vocab_sizes,
        weights=weights,
        discrete_table_stats=discrete_table_stats,
        ipca=ipca,
        rq_metrics=rq.metrics,
        opq_metrics_history=opq.metrics_history,
        rotation=opq.rotation,
        codes=codes,
        collision_summary=collision_summary,
        elapsed_seconds=time.time() - start_time,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log("Done")
    log(f"  sid_codes.npy: {codes.shape}")
    log(f"  output_dir: {args.output_dir}")
    log(f"  elapsed_seconds: {metrics['elapsed_seconds']:.1f}")
