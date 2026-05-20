#!/usr/bin/env python3
"""Evaluate SID construction quality and write zero-dependency SVG charts.

The script supports both outputs produced by:

  build_rqkmeans_sid.py
  build_rqopq_sid.py

It writes:
  summary.json
  report.md
  code_usage.csv
  prefix_stats.csv
  top_collision_groups.csv
  *.svg charts

No plotting libraries are required. Charts are generated as plain SVG.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SID quality.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("shopee_tittle_emb.npy"),
        help="Original embedding .npy used to build SID.",
    )
    parser.add_argument(
        "--sid-dir",
        type=Path,
        default=Path("rqopq_sid_out"),
        help="Directory containing SID artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory. Defaults to <sid-dir>/eval.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=8000,
        help="Maximum points in PCA scatter SVG.",
    )
    parser.add_argument(
        "--neighbor-sample-size",
        type=int,
        default=1000,
        help="Number of anchors for nearest-neighbor preservation metrics.",
    )
    parser.add_argument(
        "--neighbor-topk",
        type=int,
        default=20,
        help="Top-k neighbors used for preservation metrics.",
    )
    parser.add_argument(
        "--color-column",
        type=int,
        default=0,
        help="SID code column used to color PCA scatter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, EPS)


def entropy_from_counts(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if positive.size == 0:
        return 0.0
    prob = positive / positive.sum()
    return float(-(prob * np.log2(prob)).sum())


def gini_from_counts(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(np.float64))
    total = x.sum()
    if total <= EPS:
        return 0.0
    n = x.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x)) / (n * total) - (n + 1.0) / n)


def quantile_int(values: np.ndarray, q: float) -> int:
    if values.size == 0:
        return 0
    return int(np.quantile(values.astype(np.float64), q))


def detect_sid_layout(sid_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    if (sid_dir / "rq_codebooks.npy").exists() and (sid_dir / "opq_codebooks.npy").exists():
        rq_codebooks = np.load(sid_dir / "rq_codebooks.npy", mmap_mode="r")
        opq_codebooks = np.load(sid_dir / "opq_codebooks.npy", mmap_mode="r")
        rq_levels, rq_clusters, code_dim = rq_codebooks.shape
        opq_subspaces, opq_clusters, opq_subdim = opq_codebooks.shape
        names = [f"rq{i}" for i in range(rq_levels)] + [
            f"opq{i}" for i in range(opq_subspaces)
        ]
        clusters = [int(rq_clusters)] * int(rq_levels) + [
            int(opq_clusters)
        ] * int(opq_subspaces)
        return {
            "mode": "rqopq",
            "names": names,
            "clusters": clusters,
            "rq_levels": int(rq_levels),
            "opq_subspaces": int(opq_subspaces),
            "code_dim": int(code_dim),
            "opq_subdim": int(opq_subdim),
        }

    if (sid_dir / "codebooks.npy").exists():
        codebooks = np.load(sid_dir / "codebooks.npy", mmap_mode="r")
        levels, clusters, code_dim = codebooks.shape
        return {
            "mode": "rqkmeans",
            "names": [f"rq{i}" for i in range(levels)],
            "clusters": [int(clusters)] * int(levels),
            "rq_levels": int(levels),
            "opq_subspaces": 0,
            "code_dim": int(code_dim),
            "opq_subdim": 0,
        }

    config = metrics.get("config", {})
    if {"rq_levels", "rq_clusters", "opq_subspaces", "opq_clusters"} <= set(config):
        return {
            "mode": "rqopq",
            "names": [f"rq{i}" for i in range(config["rq_levels"])]
            + [f"opq{i}" for i in range(config["opq_subspaces"])],
            "clusters": [int(config["rq_clusters"])] * int(config["rq_levels"])
            + [int(config["opq_clusters"])] * int(config["opq_subspaces"]),
            "rq_levels": int(config["rq_levels"]),
            "opq_subspaces": int(config["opq_subspaces"]),
            "code_dim": int(config["code_dim"]),
            "opq_subdim": int(config["opq_subdim"]),
        }
    raise FileNotFoundError("Cannot detect SID layout from codebook files or metrics.json")


def project_original_embeddings(input_path: Path, projection_path: Path) -> np.ndarray:
    raw = np.load(input_path).astype(np.float32, copy=False)
    projection = np.load(projection_path)
    mean = projection["mean"].astype(np.float32)
    components = projection["components"].astype(np.float32)
    normalized = bool(projection["normalized"]) if "normalized" in projection else True
    x = l2_normalize(raw) if normalized else raw
    return ((x - mean) @ components.T).astype(np.float32)


def collision_details(codes: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping: dict[tuple[int, ...], list[int]] = {}
    for row_id, row in enumerate(codes):
        mapping.setdefault(tuple(int(v) for v in row.tolist()), []).append(row_id)
    groups = sorted(
        (
            {"sid": "-".join(map(str, sid)), "size": len(items), "row_ids": items}
            for sid, items in mapping.items()
            if len(items) > 1
        ),
        key=lambda item: item["size"],
        reverse=True,
    )
    sizes = np.array([g["size"] for g in groups], dtype=np.int64)
    summary = {
        "unique_sid": int(len(mapping)),
        "unique_sid_ratio": float(len(mapping) / codes.shape[0]),
        "collision_groups": int(len(groups)),
        "colliding_items": int(sizes.sum()) if sizes.size else 0,
        "extra_collisions": int(codes.shape[0] - len(mapping)),
        "collision_rate_extra": float((codes.shape[0] - len(mapping)) / codes.shape[0]),
        "max_collision_group_size": int(sizes.max()) if sizes.size else 1,
        "top_collision_group_sizes": sizes[:20].astype(int).tolist(),
    }
    return summary, groups


def prefix_stats(codes: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    n_items = codes.shape[0]
    for depth in range(1, codes.shape[1] + 1):
        prefix = np.ascontiguousarray(codes[:, :depth])
        row_type = np.dtype((np.void, prefix.dtype.itemsize * prefix.shape[1]))
        packed = prefix.view(row_type).reshape(-1)
        _, counts = np.unique(packed, return_counts=True)
        collision_sizes = counts[counts > 1]
        rows.append(
            {
                "depth": int(depth),
                "unique_prefixes": int(counts.shape[0]),
                "unique_ratio": float(counts.shape[0] / n_items),
                "extra_collisions": int(n_items - counts.shape[0]),
                "collision_rate_extra": float((n_items - counts.shape[0]) / n_items),
                "collision_groups": int(collision_sizes.shape[0]),
                "max_group_size": int(collision_sizes.max()) if collision_sizes.size else 1,
            }
        )
    return rows


def code_usage_stats(
    codes: np.ndarray, names: list[str], clusters: list[int]
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows = []
    counts_by_column = []
    n_items = codes.shape[0]
    for col, (name, k) in enumerate(zip(names, clusters)):
        counts = np.bincount(codes[:, col], minlength=k).astype(np.int64)
        counts_by_column.append(counts)
        nonzero = counts[counts > 0]
        entropy = entropy_from_counts(counts)
        effective = float(2.0**entropy)
        rows.append(
            {
                "column": int(col),
                "name": name,
                "clusters": int(k),
                "used_codes": int(nonzero.size),
                "cur": float(nonzero.size / k),
                "entropy": entropy,
                "entropy_ratio": float(entropy / math.log2(k)) if k > 1 else 0.0,
                "effective_codes": effective,
                "effective_code_ratio": float(effective / k),
                "gini": gini_from_counts(counts),
                "min_count_nonzero": int(nonzero.min()) if nonzero.size else 0,
                "p50_count_nonzero": quantile_int(nonzero, 0.50),
                "p90_count_nonzero": quantile_int(nonzero, 0.90),
                "p95_count_nonzero": quantile_int(nonzero, 0.95),
                "p99_count_nonzero": quantile_int(nonzero, 0.99),
                "max_count": int(counts.max()) if counts.size else 0,
                "max_share": float(counts.max() / n_items) if counts.size else 0.0,
            }
        )
    return rows, counts_by_column


def longest_common_prefix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    matches = a == b
    return np.cumprod(matches.astype(np.int8), axis=1).sum(axis=1)


def topk_neighbors(
    query: np.ndarray,
    database: np.ndarray,
    query_indices: np.ndarray,
    topk: int,
    batch_size: int = 128,
) -> np.ndarray:
    result = np.empty((query.shape[0], topk), dtype=np.int32)
    for start in range(0, query.shape[0], batch_size):
        end = min(start + batch_size, query.shape[0])
        sims = query[start:end] @ database.T
        for local, global_idx in enumerate(query_indices[start:end]):
            sims[local, int(global_idx)] = -np.inf
        candidates = np.argpartition(sims, -topk, axis=1)[:, -topk:]
        scores = np.take_along_axis(sims, candidates, axis=1)
        order = np.argsort(scores, axis=1)[:, ::-1]
        result[start:end] = np.take_along_axis(candidates, order, axis=1).astype(np.int32)
    return result


def neighbor_quality(
    projected: np.ndarray,
    reconstruction: np.ndarray | None,
    codes: np.ndarray,
    sample_size: int,
    topk: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n_items = codes.shape[0]
    sample_size = min(sample_size, n_items)
    anchors = np.sort(rng.choice(n_items, size=sample_size, replace=False)).astype(np.int32)

    original_norm = l2_normalize(projected.astype(np.float32))
    original_nn = topk_neighbors(original_norm[anchors], original_norm, anchors, topk)

    data: dict[str, np.ndarray] = {}
    flat_anchor = np.repeat(anchors, topk)
    flat_neighbor = original_nn.reshape(-1)
    data["neighbor_lcp"] = longest_common_prefix(codes[flat_anchor], codes[flat_neighbor])

    random_neighbors = rng.integers(0, n_items, size=flat_anchor.shape[0], dtype=np.int32)
    same = random_neighbors == flat_anchor
    random_neighbors[same] = (random_neighbors[same] + 1) % n_items
    data["random_lcp"] = longest_common_prefix(codes[flat_anchor], codes[random_neighbors])

    summary: dict[str, Any] = {
        "sample_size": int(sample_size),
        "topk": int(topk),
        "mean_neighbor_lcp": float(data["neighbor_lcp"].mean()),
        "mean_random_lcp": float(data["random_lcp"].mean()),
        "neighbor_lcp_hist": np.bincount(
            data["neighbor_lcp"], minlength=codes.shape[1] + 1
        ).astype(int).tolist(),
        "random_lcp_hist": np.bincount(
            data["random_lcp"], minlength=codes.shape[1] + 1
        ).astype(int).tolist(),
    }

    if reconstruction is not None:
        recon_norm = l2_normalize(reconstruction.astype(np.float32))
        recon_nn = topk_neighbors(recon_norm[anchors], recon_norm, anchors, topk)
        recalls = []
        for original_row, recon_row in zip(original_nn, recon_nn):
            recalls.append(len(set(original_row.tolist()) & set(recon_row.tolist())) / topk)
        data["nn_recall_per_anchor"] = np.array(recalls, dtype=np.float32)
        summary["reconstruction_nn_recall_at_k"] = float(data["nn_recall_per_anchor"].mean())
        summary["reconstruction_nn_recall_p50"] = float(
            np.quantile(data["nn_recall_per_anchor"], 0.5)
        )
        summary["reconstruction_nn_recall_p90"] = float(
            np.quantile(data["nn_recall_per_anchor"], 0.9)
        )

    return summary, data


def make_projected_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[:2]].astype(np.float32)
    xy = centered @ basis
    return xy.astype(np.float32), basis


def transform_2d(x: np.ndarray, reference: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = x - reference.mean(axis=0, keepdims=True)
    return (centered @ basis).astype(np.float32)


def color_for_code(code: int) -> str:
    hue = (int(code) * 137.508) % 360.0
    return f"hsl({hue:.1f},68%,48%)"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;font-size:12px;fill:#172033}",
        ".title{font-size:18px;font-weight:700}",
        ".subtitle{font-size:12px;fill:#536173}",
        ".axis{stroke:#8a96a8;stroke-width:1}",
        ".grid{stroke:#d9dee7;stroke-width:1}",
        ".bar{fill:#4f8cc9}",
        ".bar2{fill:#e58b47}",
        ".line{fill:none;stroke-width:2.2}",
        "</style>",
    ]


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.write_text("\n".join(svg_header(width, height) + body + ["</svg>"]))


def nice_range(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < EPS:
        return min(0.0, lo), hi + 1.0
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def plot_bar_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 980, 480
    left, right, top, bottom = 70, 30, 70, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    labels = [r["name"] for r in rows]
    metrics = [
        ("CUR", [r["cur"] for r in rows], "#3d7fc1"),
        ("Entropy ratio", [r["entropy_ratio"] for r in rows], "#e58b47"),
        ("Effective code ratio", [r["effective_code_ratio"] for r in rows], "#5ca66b"),
    ]
    n = len(rows)
    group_w = plot_w / max(n, 1)
    bar_w = min(36, group_w / (len(metrics) + 1))
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">Codebook Usage Quality</text>',
        f'<text class="subtitle" x="{left}" y="52">Higher is better. Values are ratios in [0, 1].</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = 1 - i / 5
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.1f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    for j, (metric_name, values, color) in enumerate(metrics):
        lx = left + j * 150
        body.append(f'<rect x="{lx}" y="{height-28}" width="12" height="12" fill="{color}"/>')
        body.append(f'<text x="{lx+18}" y="{height-18}">{esc(metric_name)}</text>')
        for i, value in enumerate(values):
            x = left + i * group_w + group_w / 2 - (len(metrics) * bar_w) / 2 + j * bar_w
            h = max(0.0, min(1.0, float(value))) * plot_h
            y = top + plot_h - h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-3:.1f}" height="{h:.1f}" fill="{color}"/>'
            )
    for i, label in enumerate(labels):
        x = left + i * group_w + group_w / 2
        body.append(
            f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle">{esc(label)}</text>'
        )
    write_svg(path, width, height, body)


def plot_usage_sorted(path: Path, names: list[str], counts_by_column: list[np.ndarray]) -> None:
    width, height = 980, 520
    left, right, top, bottom = 70, 170, 70, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    sorted_counts = [np.sort(c.astype(np.float64))[::-1] for c in counts_by_column]
    ymax = max(float(c.max()) for c in sorted_counts) if sorted_counts else 1.0
    colors = ["#3d7fc1", "#e58b47", "#5ca66b", "#b45cc7", "#d65f5f", "#6f8f3a"]
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">Sorted Code Usage Frequency</text>',
        f'<text class="subtitle" x="{left}" y="52">A steep curve means a few codes dominate the column.</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = ymax * (1 - i / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.0f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    for idx, counts in enumerate(sorted_counts):
        color = colors[idx % len(colors)]
        points = []
        k = counts.size
        step = max(1, k // 300)
        for rank in range(0, k, step):
            x = left + plot_w * rank / max(k - 1, 1)
            y = top + plot_h * (1 - counts[rank] / ymax)
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            body.append(f'<polyline class="line" stroke="{color}" points="{" ".join(points)}"/>')
        ly = top + idx * 22
        body.append(f'<line x1="{left+plot_w+24}" y1="{ly}" x2="{left+plot_w+48}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{left+plot_w+56}" y="{ly+4}">{esc(names[idx])}</text>')
    body.append(f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle">code rank by frequency</text>')
    write_svg(path, width, height, body)


def plot_residual_curve(path: Path, labels: list[str], values: list[float]) -> None:
    width, height = 860, 460
    left, right, top, bottom = 85, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    _, ymax = nice_range([0.0] + values)
    ymin = 0.0
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">Residual MSE Curve</text>',
        f'<text class="subtitle" x="{left}" y="52">Lower is better. Each stage should reduce quantization error.</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = ymax * (1 - i / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.4f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    points = []
    for i, val in enumerate(values):
        x = left + plot_w * i / max(len(values) - 1, 1)
        y = top + plot_h * (1 - (val - ymin) / max(ymax - ymin, EPS))
        points.append((x, y))
    body.append(
        '<polyline class="line" stroke="#3d7fc1" points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        + '"/>'
    )
    for (x, y), label, val in zip(points, labels, values):
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#3d7fc1"/>')
        body.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle">{esc(label)}</text>')
        body.append(f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle">{val:.4f}</text>')
    write_svg(path, width, height, body)


def plot_histogram(
    path: Path,
    values: np.ndarray,
    title: str,
    subtitle: str,
    bins: int = 40,
    x_range: tuple[float, float] | None = None,
) -> None:
    width, height = 860, 460
    left, right, top, bottom = 75, 35, 70, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = np.array([0.0])
    hist, edges = np.histogram(values, bins=bins, range=x_range)
    ymax = max(int(hist.max()), 1)
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">{esc(title)}</text>',
        f'<text class="subtitle" x="{left}" y="52">{esc(subtitle)}</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = ymax * (1 - i / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.0f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    bar_w = plot_w / len(hist)
    for i, count in enumerate(hist):
        h = plot_h * count / ymax
        x = left + i * bar_w
        y = top + plot_h - h
        body.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{max(bar_w-1,1):.1f}" height="{h:.1f}"/>')
    body.append(f'<text x="{left}" y="{top+plot_h+28}">{edges[0]:.3g}</text>')
    body.append(f'<text x="{left+plot_w}" y="{top+plot_h+28}" text-anchor="end">{edges[-1]:.3g}</text>')
    write_svg(path, width, height, body)


def plot_prefix_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 860, 460
    left, right, top, bottom = 75, 40, 70, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    depths = [r["depth"] for r in rows]
    unique = [r["unique_ratio"] for r in rows]
    collisions = [r["collision_rate_extra"] for r in rows]
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">Prefix Uniqueness by SID Depth</text>',
        f'<text class="subtitle" x="{left}" y="52">Unique ratio should increase as more SID tokens are used.</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = 1 - i / 5
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.1f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')
    for series, color, label in [
        (unique, "#3d7fc1", "unique ratio"),
        (collisions, "#d65f5f", "collision rate"),
    ]:
        points = []
        for i, val in enumerate(series):
            x = left + plot_w * i / max(len(series) - 1, 1)
            y = top + plot_h * (1 - val)
            points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<polyline class="line" stroke="{color}" points="{" ".join(points)}"/>')
        lx = left + (0 if label.startswith("unique") else 150)
        body.append(f'<line x1="{lx}" y1="{height-26}" x2="{lx+22}" y2="{height-26}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{lx+28}" y="{height-22}">{esc(label)}</text>')
    for i, depth in enumerate(depths):
        x = left + plot_w * i / max(len(depths) - 1, 1)
        body.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle">{depth}</text>')
    write_svg(path, width, height, body)


def plot_lcp_histogram(
    path: Path,
    neighbor_lcp: np.ndarray,
    random_lcp: np.ndarray,
    max_depth: int,
) -> None:
    width, height = 880, 460
    left, right, top, bottom = 75, 40, 70, 75
    plot_w = width - left - right
    plot_h = height - top - bottom
    bins = np.arange(max_depth + 1)
    h1 = np.bincount(neighbor_lcp, minlength=max_depth + 1).astype(np.float64)
    h2 = np.bincount(random_lcp, minlength=max_depth + 1).astype(np.float64)
    h1 = h1 / max(h1.sum(), EPS)
    h2 = h2 / max(h2.sum(), EPS)
    ymax = max(float(h1.max()), float(h2.max()), 0.01)
    group_w = plot_w / (max_depth + 1)
    bar_w = group_w * 0.36
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">SID Prefix Agreement for Nearest Neighbors</text>',
        f'<text class="subtitle" x="{left}" y="52">Neighbors should share longer prefixes than random pairs.</text>',
    ]
    for i in range(6):
        y = top + plot_h * i / 5
        val = ymax * (1 - i / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end">{val:.2f}</text>')
    body.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>')
    for i in bins:
        cx = left + group_w * i + group_w / 2
        for offset, hist, color in [(-bar_w / 2, h1, "#3d7fc1"), (bar_w / 2, h2, "#d65f5f")]:
            height_i = plot_h * hist[i] / ymax
            body.append(
                f'<rect x="{cx+offset-bar_w/2:.1f}" y="{top+plot_h-height_i:.1f}" '
                f'width="{bar_w:.1f}" height="{height_i:.1f}" fill="{color}"/>'
            )
        body.append(f'<text x="{cx:.1f}" y="{top+plot_h+24}" text-anchor="middle">{i}</text>')
    body.append(f'<rect x="{left}" y="{height-30}" width="12" height="12" fill="#3d7fc1"/>')
    body.append(f'<text x="{left+18}" y="{height-20}">original nearest neighbors</text>')
    body.append(f'<rect x="{left+230}" y="{height-30}" width="12" height="12" fill="#d65f5f"/>')
    body.append(f'<text x="{left+248}" y="{height-20}">random pairs</text>')
    write_svg(path, width, height, body)


def scale_points(
    xy: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    sx = width / max(xmax - xmin, EPS)
    sy = height / max(ymax - ymin, EPS)
    scale = min(sx, sy)
    px = x0 + (xy[:, 0] - xmin) * scale + (width - (xmax - xmin) * scale) / 2
    py = y0 + height - ((xy[:, 1] - ymin) * scale + (height - (ymax - ymin) * scale) / 2)
    return np.stack([px, py], axis=1)


def plot_pca_scatter(
    path: Path,
    projected: np.ndarray,
    reconstruction: np.ndarray | None,
    codes: np.ndarray,
    color_column: int,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n_items = projected.shape[0]
    sample_size = min(max_points, n_items)
    sample = np.sort(rng.choice(n_items, size=sample_size, replace=False))
    xy, basis = make_projected_2d(projected)
    xy_sample = xy[sample]

    if reconstruction is not None:
        recon_xy = transform_2d(reconstruction, projected, basis)[sample]
        all_xy = np.concatenate([xy_sample, recon_xy], axis=0)
        panels = [("original projected embedding", xy_sample), ("SID reconstruction", recon_xy)]
    else:
        all_xy = xy_sample
        panels = [("projected embedding", xy_sample)]

    xmin, xmax = np.quantile(all_xy[:, 0], [0.01, 0.99])
    ymin, ymax = np.quantile(all_xy[:, 1], [0.01, 0.99])
    xpad = 0.08 * max(xmax - xmin, EPS)
    ypad = 0.08 * max(ymax - ymin, EPS)
    bounds = (float(xmin - xpad), float(xmax + xpad), float(ymin - ypad), float(ymax + ypad))

    width, height = 1180, 560
    left, top = 55, 80
    panel_w = 500 if reconstruction is not None else 1040
    panel_h = 400
    gap = 50
    body = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">PCA-2D SID Structure View</text>',
        f'<text class="subtitle" x="{left}" y="52">Colored by SID column {color_column}. This is a lightweight PCA substitute for UMAP.</text>',
    ]
    for panel_idx, (label, panel_xy) in enumerate(panels):
        x0 = left + panel_idx * (panel_w + gap)
        y0 = top
        body.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#ccd4df"/>')
        body.append(f'<text x="{x0}" y="{y0-10}">{esc(label)}</text>')
        scaled = scale_points(panel_xy, x0, y0, panel_w, panel_h, bounds)
        for (px, py), code in zip(scaled, codes[sample, color_column]):
            body.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.35" '
                f'fill="{color_for_code(int(code))}" fill-opacity="0.55"/>'
            )
    body.append(f'<text x="{left}" y="{height-24}">sampled points: {sample_size} / {n_items}</text>')
    write_svg(path, width, height, body)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def residual_curve_from_metrics(
    metrics: dict[str, Any], projected: np.ndarray
) -> tuple[list[str], list[float]]:
    labels = ["input"]
    values = [float(np.mean(np.sum(projected * projected, axis=1)))]
    for idx, row in enumerate(metrics.get("rq_kmeans", {}).get("per_level", []), start=1):
        labels.append(f"rq{idx}")
        values.append(float(row.get("residual_mse_after_level", row.get("inertia", 0.0))))
    reconstruction = metrics.get("reconstruction", {})
    if "opq_residual_mse" in reconstruction:
        labels.append("opq")
        values.append(float(reconstruction["opq_residual_mse"]))
    return labels, values


def write_collision_csv(path: Path, groups: list[dict[str, Any]], topn: int = 100) -> None:
    rows = []
    for group in groups[:topn]:
        rows.append(
            {
                "sid": group["sid"],
                "size": group["size"],
                "row_ids_head": " ".join(map(str, group["row_ids"][:30])),
            }
        )
    write_csv(path, rows)


def write_report(
    path: Path,
    sid_dir: Path,
    output_dir: Path,
    layout: dict[str, Any],
    summary: dict[str, Any],
    chart_files: list[str],
) -> None:
    lines = [
        "# SID Quality Report",
        "",
        f"- SID dir: `{sid_dir}`",
        f"- Mode: `{layout['mode']}`",
        f"- Items: `{summary['num_items']}`",
        f"- SID tokens per item: `{summary['num_tokens']}`",
        f"- Unique SID ratio: `{summary['collision']['unique_sid_ratio']:.6f}`",
        f"- Extra collision rate: `{summary['collision']['collision_rate_extra']:.6f}`",
        f"- Max collision group size: `{summary['collision']['max_collision_group_size']}`",
        f"- Dense concat dim: `{summary.get('dense_concat_dim', 'NA')}`",
        f"- Dense reconstruction dim: `{summary.get('dense_reconstruction_dim', 'NA')}`",
        "",
        "## Core Metrics",
        "",
        f"- Mean reconstruction cosine: `{summary.get('mean_reconstruction_cosine', float('nan')):.6f}`",
        f"- Projected-space reconstruction MSE: `{summary.get('projected_reconstruction_mse', float('nan')):.6f}`",
        f"- Neighbor LCP mean: `{summary['neighbor_quality']['mean_neighbor_lcp']:.4f}`",
        f"- Random LCP mean: `{summary['neighbor_quality']['mean_random_lcp']:.4f}`",
    ]
    if "reconstruction_nn_recall_at_k" in summary["neighbor_quality"]:
        k = summary["neighbor_quality"]["topk"]
        lines.append(
            f"- Reconstruction NN recall@{k}: "
            f"`{summary['neighbor_quality']['reconstruction_nn_recall_at_k']:.6f}`"
        )
    lines += [
        "",
        "## Charts",
        "",
    ]
    for chart in chart_files:
        lines.append(f"![{chart}]({chart})")
        lines.append("")
    lines += [
        "## Data Files",
        "",
        "- `summary.json`",
        "- `code_usage.csv`",
        "- `prefix_stats.csv`",
        "- `top_collision_groups.csv`",
    ]
    path.write_text("\n".join(lines))


def write_rich_png_charts(
    output_dir: Path,
    usage_rows: list[dict[str, Any]],
    counts_by_column: list[np.ndarray],
    prefix_rows: list[dict[str, Any]],
    collision_groups: list[dict[str, Any]],
    residual_labels: list[str],
    residual_values: list[float],
    cosine_values: np.ndarray,
    neighbor_arrays: dict[str, np.ndarray],
    projected: np.ndarray,
    reconstruction: np.ndarray | None,
    codes: np.ndarray,
    color_column: int,
    max_scatter_points: int,
    seed: int,
) -> list[str]:
    """Write richer PNG plots when matplotlib/sklearn are installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except Exception as exc:
        (output_dir / "rich_plot_error.txt").write_text(
            f"Rich PNG plots skipped because plotting dependencies failed: {exc}\n"
        )
        return []

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180

    chart_files: list[str] = []

    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, bbox_inches="tight")
        plt.close()
        chart_files.append(name)

    names = [row["name"] for row in usage_rows]
    x = np.arange(len(names))

    plt.figure(figsize=(10.5, 5.2))
    width = 0.25
    plt.bar(x - width, [row["cur"] for row in usage_rows], width, label="CUR")
    plt.bar(x, [row["entropy_ratio"] for row in usage_rows], width, label="Entropy ratio")
    plt.bar(
        x + width,
        [row["effective_code_ratio"] for row in usage_rows],
        width,
        label="Effective code ratio",
    )
    plt.xticks(x, names)
    plt.ylim(0, 1.08)
    plt.title("Codebook Usage Quality")
    plt.ylabel("ratio")
    plt.legend(loc="lower right")
    save_current("rich_01_codebook_usage_quality.png")

    plt.figure(figsize=(10.5, 5.2))
    for name, counts in zip(names, counts_by_column):
        sorted_counts = np.sort(counts.astype(np.float64))[::-1]
        plt.plot(np.arange(1, len(sorted_counts) + 1), sorted_counts, label=name, linewidth=2)
    plt.yscale("log")
    plt.title("Sorted Code Usage Frequency")
    plt.xlabel("code rank by frequency")
    plt.ylabel("item count, log scale")
    plt.legend(ncol=2)
    save_current("rich_02_sorted_code_usage_frequency.png")

    if len(residual_values) > 1:
        plt.figure(figsize=(8.5, 5.0))
        plt.plot(residual_labels, residual_values, marker="o", linewidth=2.5)
        for label, value in zip(residual_labels, residual_values):
            plt.text(label, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        plt.title("Residual MSE Curve")
        plt.ylabel("MSE")
        save_current("rich_03_residual_mse_curve.png")

    if cosine_values.size:
        plt.figure(figsize=(8.5, 5.0))
        sns.histplot(cosine_values, bins=60, kde=True)
        plt.title("Projected Embedding vs SID Reconstruction Cosine")
        plt.xlabel("cosine similarity")
        save_current("rich_04_reconstruction_cosine_histogram.png")

    collision_sizes = np.array([group["size"] for group in collision_groups], dtype=np.float32)
    if collision_sizes.size:
        plt.figure(figsize=(8.5, 5.0))
        sns.histplot(collision_sizes, bins=np.arange(1.5, collision_sizes.max() + 2.5, 1.0))
        plt.title("Collision Group Size Histogram")
        plt.xlabel("group size")
        plt.ylabel("collision groups")
        save_current("rich_05_collision_group_size_histogram.png")

    plt.figure(figsize=(8.5, 5.0))
    depths = [row["depth"] for row in prefix_rows]
    plt.plot(
        depths,
        [row["unique_ratio"] for row in prefix_rows],
        marker="o",
        linewidth=2.5,
        label="unique ratio",
    )
    plt.plot(
        depths,
        [row["collision_rate_extra"] for row in prefix_rows],
        marker="o",
        linewidth=2.5,
        label="collision rate",
    )
    plt.xticks(depths)
    plt.ylim(0, 1.05)
    plt.title("Prefix Uniqueness by SID Depth")
    plt.xlabel("SID depth")
    plt.ylabel("ratio")
    plt.legend()
    save_current("rich_06_prefix_uniqueness_curve.png")

    plt.figure(figsize=(8.5, 5.0))
    max_depth = codes.shape[1]
    bins = np.arange(max_depth + 1)
    neighbor_hist = np.bincount(
        neighbor_arrays["neighbor_lcp"], minlength=max_depth + 1
    ).astype(np.float64)
    random_hist = np.bincount(
        neighbor_arrays["random_lcp"], minlength=max_depth + 1
    ).astype(np.float64)
    neighbor_hist /= max(neighbor_hist.sum(), EPS)
    random_hist /= max(random_hist.sum(), EPS)
    bar_width = 0.38
    plt.bar(bins - bar_width / 2, neighbor_hist, bar_width, label="nearest neighbors")
    plt.bar(bins + bar_width / 2, random_hist, bar_width, label="random pairs")
    plt.xticks(bins)
    plt.title("SID Prefix Agreement")
    plt.xlabel("longest common prefix length")
    plt.ylabel("pair share")
    plt.legend()
    save_current("rich_07_neighbor_prefix_agreement.png")

    if "nn_recall_per_anchor" in neighbor_arrays:
        plt.figure(figsize=(8.5, 5.0))
        sns.histplot(neighbor_arrays["nn_recall_per_anchor"], bins=20, kde=True)
        plt.title("Nearest-Neighbor Recall of SID Reconstruction")
        plt.xlabel("recall@k per sampled anchor")
        save_current("rich_08_reconstruction_nn_recall_histogram.png")

    rng = np.random.default_rng(seed)
    n_items = projected.shape[0]
    scatter_points = min(max_scatter_points, 1500 if reconstruction is not None else 2500, n_items)
    sample = np.sort(rng.choice(n_items, size=scatter_points, replace=False))
    panels = [("original projected embedding", projected[sample])]
    if reconstruction is not None:
        panels.append(("SID reconstruction", reconstruction[sample]))
    combined = np.concatenate([panel[1] for panel in panels], axis=0).astype(np.float32)

    method = "t-SNE"
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.08,
            metric="cosine",
            random_state=seed,
        )
        xy_all = reducer.fit_transform(combined)
        method = "UMAP"
    except Exception:
        try:
            pre_dim = min(20, combined.shape[1], combined.shape[0] - 1)
            pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(combined)
            perplexity = min(30, max(5, (combined.shape[0] - 1) // 3))
            xy_all = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                max_iter=750,
                random_state=seed,
            ).fit_transform(pre)
        except Exception:
            method = "PCA"
            xy_all = PCA(n_components=2, random_state=seed).fit_transform(combined)

    plt.figure(figsize=(12.0, 5.5))
    offset = 0
    color_values = codes[sample, color_column]
    color_base = max(int(color_values.max()) + 1, 1)
    colors = (color_values % color_base) / color_base
    for idx, (title, data) in enumerate(panels, start=1):
        xy = xy_all[offset : offset + data.shape[0]]
        offset += data.shape[0]
        ax = plt.subplot(1, len(panels), idx)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=colors,
            cmap="hsv",
            s=7,
            alpha=0.62,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(f"{method} SID Structure View, colored by SID column {color_column}")
    save_current(f"rich_09_{method.lower()}_sid_structure.png")

    return chart_files


def main() -> None:
    args = parse_args()
    sid_dir = args.sid_dir
    output_dir = args.output_dir or sid_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Loading SID artifacts from {sid_dir}")
    metrics = read_json(sid_dir / "metrics.json")
    layout = detect_sid_layout(sid_dir, metrics)
    codes = np.load(sid_dir / "sid_codes.npy").astype(np.int32, copy=False)
    reconstruction = (
        np.load(sid_dir / "sid_reconstruction.npy").astype(np.float32, copy=False)
        if (sid_dir / "sid_reconstruction.npy").exists()
        else None
    )
    dense_concat_dim = None
    if (sid_dir / "sid_codeword_concat.npy").exists():
        dense_concat_dim = int(np.load(sid_dir / "sid_codeword_concat.npy", mmap_mode="r").shape[1])

    log("Recomputing projected input embedding")
    projected = project_original_embeddings(args.input, sid_dir / "projection.npz")

    log("Computing code usage and collision metrics")
    usage_rows, counts_by_column = code_usage_stats(
        codes, layout["names"], layout["clusters"]
    )
    prefix_rows = prefix_stats(codes)
    collision_summary, collision_groups = collision_details(codes)

    projected_reconstruction_mse = None
    cosine_values = np.array([], dtype=np.float32)
    if reconstruction is not None:
        residual = projected - reconstruction
        projected_reconstruction_mse = float(np.mean(np.sum(residual * residual, axis=1)))
        cosine_values = np.sum(projected * reconstruction, axis=1) / np.maximum(
            np.linalg.norm(projected, axis=1) * np.linalg.norm(reconstruction, axis=1),
            EPS,
        )

    log("Computing nearest-neighbor preservation metrics")
    neighbor_summary, neighbor_arrays = neighbor_quality(
        projected=projected,
        reconstruction=reconstruction,
        codes=codes,
        sample_size=args.neighbor_sample_size,
        topk=args.neighbor_topk,
        seed=args.seed,
    )

    summary: dict[str, Any] = {
        "sid_dir": str(sid_dir),
        "mode": layout["mode"],
        "num_items": int(codes.shape[0]),
        "num_tokens": int(codes.shape[1]),
        "layout": layout,
        "dense_concat_dim": dense_concat_dim,
        "dense_reconstruction_dim": int(reconstruction.shape[1])
        if reconstruction is not None
        else None,
        "collision": collision_summary,
        "code_usage": usage_rows,
        "prefix_stats": prefix_rows,
        "neighbor_quality": neighbor_summary,
    }
    if projected_reconstruction_mse is not None:
        summary["projected_reconstruction_mse"] = projected_reconstruction_mse
        summary["mean_reconstruction_cosine"] = float(np.mean(cosine_values))
        summary["p50_reconstruction_cosine"] = float(np.quantile(cosine_values, 0.5))
        summary["p10_reconstruction_cosine"] = float(np.quantile(cosine_values, 0.1))
        summary["p90_reconstruction_cosine"] = float(np.quantile(cosine_values, 0.9))

    write_csv(output_dir / "code_usage.csv", usage_rows)
    write_csv(output_dir / "prefix_stats.csv", prefix_rows)
    write_collision_csv(output_dir / "top_collision_groups.csv", collision_groups)
    save_json(output_dir / "summary.json", summary)

    log("Writing SVG charts")
    chart_files: list[str] = []
    chart_files.append("01_codebook_usage_quality.svg")
    plot_bar_metrics(output_dir / chart_files[-1], usage_rows)

    chart_files.append("02_sorted_code_usage_frequency.svg")
    plot_usage_sorted(output_dir / chart_files[-1], layout["names"], counts_by_column)

    labels, values = residual_curve_from_metrics(metrics, projected)
    if len(values) > 1:
        chart_files.append("03_residual_mse_curve.svg")
        plot_residual_curve(output_dir / chart_files[-1], labels, values)

    if cosine_values.size:
        chart_files.append("04_reconstruction_cosine_histogram.svg")
        plot_histogram(
            output_dir / chart_files[-1],
            cosine_values,
            "Projected Embedding vs SID Reconstruction Cosine",
            "Higher is better. Values near 1 mean the quantized feature preserves direction.",
            bins=50,
            x_range=(-1.0, 1.0),
        )

    collision_sizes = np.array([g["size"] for g in collision_groups], dtype=np.float32)
    if collision_sizes.size:
        chart_files.append("05_collision_group_size_histogram.svg")
        plot_histogram(
            output_dir / chart_files[-1],
            collision_sizes,
            "Collision Group Size Histogram",
            "Counts only SID groups with at least two items.",
            bins=max(3, min(30, int(collision_sizes.max()))),
        )

    chart_files.append("06_prefix_uniqueness_curve.svg")
    plot_prefix_curve(output_dir / chart_files[-1], prefix_rows)

    chart_files.append("07_neighbor_prefix_agreement.svg")
    plot_lcp_histogram(
        output_dir / chart_files[-1],
        neighbor_arrays["neighbor_lcp"],
        neighbor_arrays["random_lcp"],
        max_depth=codes.shape[1],
    )

    if "nn_recall_per_anchor" in neighbor_arrays:
        chart_files.append("08_reconstruction_nn_recall_histogram.svg")
        plot_histogram(
            output_dir / chart_files[-1],
            neighbor_arrays["nn_recall_per_anchor"],
            "Nearest-Neighbor Recall of SID Reconstruction",
            "For each sampled anchor, overlap between original and reconstructed top-k neighbors.",
            bins=20,
            x_range=(0.0, 1.0),
        )

    color_column = min(max(args.color_column, 0), codes.shape[1] - 1)
    chart_files.append("09_pca_sid_structure.svg")
    plot_pca_scatter(
        output_dir / chart_files[-1],
        projected=projected,
        reconstruction=reconstruction,
        codes=codes,
        color_column=color_column,
        max_points=args.max_scatter_points,
        seed=args.seed,
    )

    log("Writing rich PNG charts")
    rich_chart_files = write_rich_png_charts(
        output_dir=output_dir,
        usage_rows=usage_rows,
        counts_by_column=counts_by_column,
        prefix_rows=prefix_rows,
        collision_groups=collision_groups,
        residual_labels=labels,
        residual_values=values,
        cosine_values=cosine_values,
        neighbor_arrays=neighbor_arrays,
        projected=projected,
        reconstruction=reconstruction,
        codes=codes,
        color_column=color_column,
        max_scatter_points=args.max_scatter_points,
        seed=args.seed,
    )
    chart_files.extend(rich_chart_files)

    write_report(
        output_dir / "report.md",
        sid_dir=sid_dir,
        output_dir=output_dir,
        layout=layout,
        summary=summary,
        chart_files=chart_files,
    )

    log("Done")
    log(f"  report: {output_dir / 'report.md'}")
    log(f"  summary: {output_dir / 'summary.json'}")
    log(f"  charts: {len(chart_files)} files")
    log(
        "  unique_sid_ratio="
        f"{collision_summary['unique_sid_ratio']:.6f}, "
        f"collision_rate={collision_summary['collision_rate_extra']:.6f}"
    )
    if "reconstruction_nn_recall_at_k" in neighbor_summary:
        log(
            f"  reconstruction_nn_recall@{neighbor_summary['topk']}="
            f"{neighbor_summary['reconstruction_nn_recall_at_k']:.6f}"
        )


if __name__ == "__main__":
    main()
