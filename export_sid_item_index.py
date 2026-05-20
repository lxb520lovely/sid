#!/usr/bin/env python3
"""Export SID-to-item lookup tables for inspecting SID collisions.

Given an output directory produced by build_multimodal_rqopq_sid.py, this script
joins sid_codes.npy with itemid.npy and writes:

  item_sid_index.csv       one row per item
  sid_groups_summary.csv   one row per SID group
  collision_items.csv      one row per item in collision groups

It can also query a concrete SID, e.g. --query-sid 1,2,3,4,5.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SID -> item_id index tables.")
    parser.add_argument(
        "--sid-dir",
        type=Path,
        required=True,
        help="Directory containing sid_codes.npy and itemid.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <sid-dir>/sid_item_index.",
    )
    parser.add_argument(
        "--query-sid",
        type=str,
        default=None,
        help="Comma-separated SID codes to query, e.g. 12,34,56,7,8.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum group size written to collision_items.csv.",
    )
    parser.add_argument(
        "--max-items-per-group",
        type=int,
        default=200,
        help="Max item IDs stored in one sid_groups_summary.csv row.",
    )
    parser.add_argument(
        "--write-item-index",
        action="store_true",
        help="Also write item_sid_index.csv. This can be large for full data.",
    )
    return parser.parse_args()


def sid_to_str(row: np.ndarray) -> str:
    return "-".join(str(int(x)) for x in row)


def write_item_index(
    path: Path,
    item_ids: np.ndarray,
    codes: np.ndarray,
    group_sizes_by_row: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        code_cols = [f"sid_{i}" for i in range(codes.shape[1])]
        writer.writerow(["row_idx", "item_id", "sid", "group_size", *code_cols])
        for row_idx, (item_id, code, group_size) in enumerate(
            zip(item_ids, codes, group_sizes_by_row, strict=True)
        ):
            writer.writerow(
                [row_idx, int(item_id), sid_to_str(code), int(group_size), *code.tolist()]
            )


def write_group_tables(
    output_dir: Path,
    item_ids: np.ndarray,
    codes: np.ndarray,
    unique_codes: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
    min_group_size: int,
    max_items_per_group: int,
) -> None:
    group_order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[group_order]
    boundaries = np.searchsorted(
        sorted_inverse, np.arange(counts.shape[0] + 1), side="left"
    )
    order = np.argsort(-counts, kind="stable")

    summary_path = output_dir / "sid_groups_summary.csv"
    collision_path = output_dir / "collision_items.csv"

    with summary_path.open("w", newline="") as summary_file, collision_path.open(
        "w", newline=""
    ) as collision_file:
        summary_writer = csv.writer(summary_file)
        collision_writer = csv.writer(collision_file)

        code_cols = [f"sid_{i}" for i in range(codes.shape[1])]
        summary_writer.writerow(
            [
                "sid",
                "group_size",
                "item_ids_head",
                "truncated",
                *code_cols,
            ]
        )
        collision_writer.writerow(["sid", "group_size", "row_idx", "item_id", *code_cols])

        for group_id in order:
            group_size = int(counts[group_id])
            if group_size < min_group_size:
                continue

            row_indices = group_order[boundaries[group_id] : boundaries[group_id + 1]]
            group_item_ids = item_ids[row_indices]
            code = unique_codes[group_id]
            head_item_ids = group_item_ids[:max_items_per_group]
            truncated = group_size > max_items_per_group

            summary_writer.writerow(
                [
                    sid_to_str(code),
                    group_size,
                    "|".join(str(int(x)) for x in head_item_ids),
                    int(truncated),
                    *code.tolist(),
                ]
            )

            for row_idx, item_id in zip(row_indices, group_item_ids, strict=True):
                collision_writer.writerow(
                    [sid_to_str(code), group_size, int(row_idx), int(item_id), *code.tolist()]
                )


def query_sid(item_ids: np.ndarray, codes: np.ndarray, query: str) -> None:
    wanted = np.array([int(part) for part in query.split(",")], dtype=codes.dtype)
    if wanted.shape[0] != codes.shape[1]:
        raise ValueError(f"query SID has {wanted.shape[0]} parts, expected {codes.shape[1]}")
    mask = np.all(codes == wanted[None, :], axis=1)
    matched_rows = np.flatnonzero(mask)
    print(f"SID {sid_to_str(wanted)} matched {matched_rows.size} items")
    for row_idx in matched_rows:
        print(f"{int(row_idx)}\t{int(item_ids[row_idx])}")


def main() -> None:
    args = parse_args()
    sid_dir = args.sid_dir
    output_dir = args.output_dir or sid_dir / "sid_item_index"
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = np.load(sid_dir / "sid_codes.npy", mmap_mode="r")
    item_ids = np.load(sid_dir / "itemid.npy", mmap_mode="r")
    if codes.shape[0] != item_ids.shape[0]:
        raise ValueError(
            f"row mismatch: sid_codes has {codes.shape[0]} rows, itemid has {item_ids.shape[0]}"
        )

    if args.query_sid:
        query_sid(item_ids, codes, args.query_sid)
        return

    unique_codes, inverse, counts = np.unique(
        np.asarray(codes), axis=0, return_inverse=True, return_counts=True
    )
    group_sizes_by_row = counts[inverse]

    write_group_tables(
        output_dir=output_dir,
        item_ids=item_ids,
        codes=codes,
        unique_codes=unique_codes,
        inverse=inverse,
        counts=counts,
        min_group_size=args.min_group_size,
        max_items_per_group=args.max_items_per_group,
    )
    if args.write_item_index:
        write_item_index(output_dir / "item_sid_index.csv", item_ids, codes, group_sizes_by_row)

    collision_item_count = int((group_sizes_by_row >= args.min_group_size).sum())
    collision_group_count = int((counts >= args.min_group_size).sum())
    print(f"rows: {codes.shape[0]}")
    print(f"unique_sid: {unique_codes.shape[0]}")
    print(f"collision_groups: {collision_group_count}")
    print(f"collision_items: {collision_item_count}")
    print(f"wrote: {output_dir / 'sid_groups_summary.csv'}")
    print(f"wrote: {output_dir / 'collision_items.csv'}")
    if args.write_item_index:
        print(f"wrote: {output_dir / 'item_sid_index.csv'}")


if __name__ == "__main__":
    main()
