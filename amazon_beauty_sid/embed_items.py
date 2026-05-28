from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build item embeddings from Amazon item text.")
    parser.add_argument(
        "--item-text",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed/item_text.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed"),
    )
    parser.add_argument(
        "--encoder",
        choices=("hashing-svd", "sentence-transformers"),
        default="hashing-svd",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/sentence-t5-base",
        help="Used only with --encoder sentence-transformers.",
    )
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=2**18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_item_text(path: Path) -> tuple[list[int], list[str], list[str]]:
    item_ids: list[int] = []
    asins: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_ids.append(int(row["item_idx"]))
            asins.append(row["asin"])
            texts.append(row["semantic_text"])
    expected = list(range(len(item_ids)))
    if item_ids != expected:
        raise ValueError("item_text.csv must be sorted by contiguous item_idx")
    return item_ids, asins, texts


def hashing_svd_embeddings(
    texts: list[str],
    emb_dim: int,
    hash_features: int,
    seed: int,
) -> np.ndarray:
    vectorizer = HashingVectorizer(
        n_features=hash_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    x = vectorizer.transform(texts)
    if x.nnz == 0 or len(texts) < 2:
        return np.zeros((len(texts), emb_dim), dtype=np.float32)

    n_components = min(emb_dim, max(1, len(texts) - 1), hash_features - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    emb = svd.fit_transform(x).astype(np.float32)
    if emb.shape[1] == emb_dim:
        return emb
    padded = np.zeros((len(texts), emb_dim), dtype=np.float32)
    padded[:, : emb.shape[1]] = emb
    return padded


def sentence_transformer_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Use --encoder hashing-svd "
            "or install sentence-transformers in the project venv."
        ) from exc

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_ids, asins, texts = read_item_text(args.item_text)
    if args.encoder == "hashing-svd":
        embeddings = hashing_svd_embeddings(
            texts,
            emb_dim=args.emb_dim,
            hash_features=args.hash_features,
            seed=args.seed,
        )
    else:
        embeddings = sentence_transformer_embeddings(
            texts,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )

    np.save(args.output_dir / "item_embeddings.npy", embeddings)
    np.save(args.output_dir / "itemid.npy", np.asarray(item_ids, dtype=np.int64))
    summary = {
        "item_text": str(args.item_text),
        "encoder": args.encoder,
        "model_name": args.model_name if args.encoder == "sentence-transformers" else None,
        "num_items": len(item_ids),
        "embedding_shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "first_asin": asins[0] if asins else None,
    }
    (args.output_dir / "embedding_stats.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
