from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


UNK = "<UNK>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Amazon Beauty reviews/metadata for SID experiments."
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/amazon_beauty/raw/reviews_Beauty_5.json.gz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/amazon_beauty/raw/meta_Beauty.json.gz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed"),
    )
    parser.add_argument("--min-rating", type=float, default=0.0)
    parser.add_argument(
        "--min-user-items",
        type=int,
        default=5,
        help="Keep users with at least this many interactions. Amazon 5-core uses 5.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep repeated user-item rows instead of retaining the earliest one.",
    )
    parser.add_argument(
        "--item-order",
        choices=("first_seen", "asin"),
        default="first_seen",
        help="first_seen follows review-file order, matching common open-source pipelines.",
    )
    return parser.parse_args()


def parse_loose_json(line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


def read_gzip_records(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield parse_loose_json(line)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {normalize_text(v)}" for k, v in value.items())
    return str(value).replace("\n", " ").strip()


def category_path(meta: dict[str, Any]) -> list[str]:
    categories = meta.get("categories", [])
    if not isinstance(categories, list) or not categories:
        return []
    first = categories[0]
    if not isinstance(first, list):
        return []
    path = [normalize_text(x) for x in first if normalize_text(x)]
    if path and path[0].lower() == "beauty":
        path = path[1:]
    return path


def sales_rank(meta: dict[str, Any]) -> str:
    raw = meta.get("salesRank")
    if isinstance(raw, dict):
        return json.dumps(raw, sort_keys=True)
    return normalize_text(raw)


def price(meta: dict[str, Any]) -> str:
    return normalize_text(meta.get("price"))


def load_interactions(
    reviews_path: Path,
    min_rating: float,
    keep_duplicates: bool,
) -> tuple[list[tuple[str, str, float, int]], list[str]]:
    first_seen_items: list[str] = []
    seen_items: set[str] = set()

    if keep_duplicates:
        rows: list[tuple[str, str, float, int]] = []
        for rec in read_gzip_records(reviews_path):
            rating = float(rec.get("overall", 0.0))
            if rating < min_rating:
                continue
            user = normalize_text(rec.get("reviewerID"))
            asin = normalize_text(rec.get("asin"))
            if not user or not asin:
                continue
            if asin not in seen_items:
                first_seen_items.append(asin)
                seen_items.add(asin)
            rows.append((user, asin, rating, int(rec.get("unixReviewTime", 0))))
        return rows, first_seen_items

    earliest: dict[tuple[str, str], tuple[str, str, float, int]] = {}
    for rec in read_gzip_records(reviews_path):
        rating = float(rec.get("overall", 0.0))
        if rating < min_rating:
            continue
        user = normalize_text(rec.get("reviewerID"))
        asin = normalize_text(rec.get("asin"))
        if not user or not asin:
            continue
        if asin not in seen_items:
            first_seen_items.append(asin)
            seen_items.add(asin)
        timestamp = int(rec.get("unixReviewTime", 0))
        key = (user, asin)
        old = earliest.get(key)
        if old is None or timestamp < old[3]:
            earliest[key] = (user, asin, rating, timestamp)
    return list(earliest.values()), first_seen_items


def load_metadata(metadata_path: Path, wanted_asins: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for rec in read_gzip_records(metadata_path):
        asin = normalize_text(rec.get("asin"))
        if asin in wanted_asins:
            out[asin] = rec
    return out


def item_semantic_text(meta: dict[str, Any], asin: str) -> str:
    title = normalize_text(meta.get("title")) or asin
    brand = normalize_text(meta.get("brand")) or UNK
    cats = category_path(meta)
    fields = [
        f"Title: {title}",
        f"Brand: {brand}",
        f"Categories: {' > '.join(cats) if cats else UNK}",
    ]
    p = price(meta)
    if p:
        fields.append(f"Price: {p}")
    rank = sales_rank(meta)
    if rank:
        fields.append(f"SalesRank: {rank}")
    return "; ".join(fields)


def write_csv(path: Path, header: list[str], rows: Iterable[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "splits").mkdir(exist_ok=True)

    interactions, first_seen_items = load_interactions(
        args.reviews,
        min_rating=args.min_rating,
        keep_duplicates=args.keep_duplicates,
    )
    by_user: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for user, asin, rating, timestamp in interactions:
        by_user[user].append((asin, rating, timestamp))

    user_rows = []
    kept_item_set: set[str] = set()
    for user, rows in by_user.items():
        rows.sort(key=lambda x: (x[2], x[0]))
        if len(rows) >= args.min_user_items:
            user_rows.append((user, rows))
            kept_item_set.update(asin for asin, _, _ in rows)
    user_rows.sort(key=lambda x: x[0])

    if args.item_order == "first_seen":
        item_asins = [asin for asin in first_seen_items if asin in kept_item_set]
    else:
        item_asins = sorted(kept_item_set)
    asin_to_item = {asin: idx for idx, asin in enumerate(item_asins)}
    user_to_idx = {user: idx for idx, (user, _) in enumerate(user_rows)}
    metadata = load_metadata(args.metadata, set(item_asins))
    item_counts = Counter(asin for _, rows in user_rows for asin, _, _ in rows)

    item_rows: list[list[Any]] = []
    item_text_rows: list[list[Any]] = []
    for item_idx, asin in enumerate(item_asins):
        meta = metadata.get(asin, {})
        title = normalize_text(meta.get("title")) or asin
        brand = normalize_text(meta.get("brand")) or UNK
        cats = category_path(meta)
        semantic_text = item_semantic_text(meta, asin)
        item_rows.append(
            [
                item_idx,
                asin,
                title,
                brand,
                " > ".join(cats),
                price(meta),
                sales_rank(meta),
                item_counts[asin],
                int(asin in metadata),
            ]
        )
        item_text_rows.append([item_idx, asin, semantic_text])

    sequence_records = []
    train_samples = []
    valid_samples = []
    test_samples = []
    for user, rows in user_rows:
        user_idx = user_to_idx[user]
        items = [asin_to_item[asin] for asin, _, _ in rows]
        ratings = [rating for _, rating, _ in rows]
        timestamps = [timestamp for _, _, timestamp in rows]
        rec = {
            "user_idx": user_idx,
            "reviewerID": user,
            "items": items,
            "ratings": ratings,
            "timestamps": timestamps,
        }
        sequence_records.append(rec)

        train_items = items[:-2]
        for pos in range(1, len(train_items)):
            train_samples.append(
                {
                    "user_idx": user_idx,
                    "history": train_items[:pos],
                    "target": train_items[pos],
                }
            )
        valid_samples.append(
            {
                "user_idx": user_idx,
                "history": items[:-2],
                "target": items[-2],
            }
        )
        test_samples.append(
            {
                "user_idx": user_idx,
                "history": items[:-1],
                "target": items[-1],
            }
        )

    write_csv(
        args.output_dir / "items.csv",
        [
            "item_idx",
            "asin",
            "title",
            "brand",
            "category_path",
            "price",
            "sales_rank",
            "interaction_count",
            "has_metadata",
        ],
        item_rows,
    )
    write_csv(
        args.output_dir / "item_text.csv",
        ["item_idx", "asin", "semantic_text"],
        item_text_rows,
    )
    write_csv(
        args.output_dir / "users.csv",
        ["user_idx", "reviewerID", "sequence_length"],
        ([user_to_idx[user], user, len(rows)] for user, rows in user_rows),
    )
    write_csv(
        args.output_dir / "interactions.csv",
        ["user_idx", "reviewerID", "item_idx", "asin", "rating", "timestamp"],
        (
            [user_to_idx[user], user, asin_to_item[asin], asin, rating, timestamp]
            for user, rows in user_rows
            for asin, rating, timestamp in rows
        ),
    )

    for name, records in [
        ("sequences.jsonl", sequence_records),
        ("splits/train.jsonl", train_samples),
        ("splits/valid.jsonl", valid_samples),
        ("splits/test.jsonl", test_samples),
    ]:
        with (args.output_dir / name).open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    np.save(args.output_dir / "itemid.npy", np.arange(len(item_asins), dtype=np.int64))
    lengths = [len(rec["items"]) for rec in sequence_records]
    stats = {
        "reviews": str(args.reviews),
        "metadata": str(args.metadata),
        "min_rating": args.min_rating,
        "min_user_items": args.min_user_items,
        "keep_duplicates": args.keep_duplicates,
        "item_order": args.item_order,
        "num_users": len(sequence_records),
        "num_items": len(item_asins),
        "num_interactions": int(sum(lengths)),
        "num_train_samples": len(train_samples),
        "num_valid_samples": len(valid_samples),
        "num_test_samples": len(test_samples),
        "sequence_length": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
        },
        "metadata_coverage": float(len(metadata) / max(len(item_asins), 1)),
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
