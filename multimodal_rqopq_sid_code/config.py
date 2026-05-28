from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


DEFAULT_TITLE_WEIGHT = 1.0
DEFAULT_IMAGE_WEIGHT = 0.01
DEFAULT_CAT_WEIGHT = 0.20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)
