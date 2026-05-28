from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight NTP evaluation for SID tokens.")
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed/sequences.jsonl"),
    )
    parser.add_argument(
        "--sid-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/sid_rqopq"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/ntp_sid_ngram"),
    )
    parser.add_argument("--representation", choices=("sid", "item"), default="sid")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=512)
    parser.add_argument("--max-history-items", type=int, default=50)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--ks", type=str, default="5,10,20")
    parser.add_argument("--include-valid-in-train", action="store_true")
    parser.add_argument("--no-filter-seen", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--predictions-head", type=int, default=200)
    return parser.parse_args()


def read_sequences(path: Path) -> list[list[int]]:
    sequences: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            items = [int(x) for x in rec["items"]]
            if len(items) >= 3:
                sequences.append(items)
    return sequences


@dataclass
class NgramLM:
    order: int
    vocab_size: int
    alpha: float
    counts: dict[tuple[int, ...], Counter[int]] = field(default_factory=dict)
    totals: dict[tuple[int, ...], int] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        token_streams: Iterable[list[int]],
        order: int,
        vocab_size: int,
        alpha: float,
    ) -> "NgramLM":
        counts: defaultdict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for tokens in token_streams:
            for pos in range(1, len(tokens)):
                target = tokens[pos]
                max_ctx = min(order - 1, pos)
                for ctx_len in range(max_ctx + 1):
                    ctx = tuple(tokens[pos - ctx_len : pos]) if ctx_len else ()
                    counts[ctx][target] += 1
        frozen = dict(counts)
        totals = {ctx: sum(counter.values()) for ctx, counter in frozen.items()}
        return cls(order=order, vocab_size=vocab_size, alpha=alpha, counts=frozen, totals=totals)

    def logprob(self, context: list[int], token: int) -> float:
        max_ctx = min(self.order - 1, len(context))
        for ctx_len in range(max_ctx, -1, -1):
            ctx = tuple(context[-ctx_len:]) if ctx_len else ()
            counter = self.counts.get(ctx)
            if not counter:
                continue
            count = counter.get(token, 0)
            if count > 0 or ctx_len == 0:
                total = self.totals[ctx]
                return math.log((count + self.alpha) / (total + self.alpha * self.vocab_size))
        return -math.log(self.vocab_size)


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    items: list[int] = field(default_factory=list)


def build_trie(item_tokens: list[list[int]], popularity: Counter[int]) -> TrieNode:
    root = TrieNode()
    for item, tokens in enumerate(item_tokens):
        node = root
        for token in tokens:
            node = node.children.setdefault(token, TrieNode())
        node.items.append(item)

    def sort_leaf_items(node: TrieNode) -> None:
        if node.items:
            node.items.sort(key=lambda item: (-popularity[item], item))
        for child in node.children.values():
            sort_leaf_items(child)

    sort_leaf_items(root)
    return root


def load_item_tokens(
    representation: str,
    sid_dir: Path,
    sequences: list[list[int]],
) -> tuple[list[list[int]], int, int, int]:
    num_items = max(max(seq) for seq in sequences) + 1
    if representation == "item":
        sep = num_items
        bos = num_items + 1
        return [[item] for item in range(num_items)], sep, bos, num_items + 2

    codes = np.load(sid_dir / "sid_codes_offset.npy", mmap_mode="r")
    item_ids = np.load(sid_dir / "itemid.npy", mmap_mode="r")
    if codes.shape[0] != item_ids.shape[0]:
        raise ValueError("sid_codes_offset.npy and itemid.npy row counts differ")
    if not np.array_equal(np.asarray(item_ids, dtype=np.int64), np.arange(codes.shape[0])):
        raise ValueError("This evaluator expects contiguous itemid.npy values")
    if codes.shape[0] < num_items:
        raise ValueError("SID directory has fewer rows than sequence items")
    max_sid_token = int(np.max(codes[:num_items]))
    sep = max_sid_token + 1
    bos = max_sid_token + 2
    item_tokens = [np.asarray(codes[item], dtype=np.int64).astype(int).tolist() for item in range(num_items)]
    return item_tokens, sep, bos, bos + 1


def encode_items(items: list[int], item_tokens: list[list[int]], sep: int, bos: int) -> list[int]:
    tokens = [bos]
    for item in items:
        tokens.extend(item_tokens[item])
        tokens.append(sep)
    return tokens


def topk_decode(
    lm: NgramLM,
    trie: TrieNode,
    history: list[int],
    item_tokens: list[list[int]],
    sep: int,
    bos: int,
    popularity_items: list[int],
    top_k: int,
    beam_size: int,
    max_history_items: int,
    filter_seen: bool,
) -> list[int]:
    history_tail = history[-max_history_items:] if max_history_items > 0 else history
    context = encode_items(history_tail, item_tokens, sep, bos)
    beams: list[tuple[float, int, TrieNode, list[int]]] = [(0.0, 0, trie, context)]
    uid = 1

    for _ in range(len(item_tokens[0])):
        next_beams: list[tuple[float, int, TrieNode, list[int]]] = []
        for score, _, node, ctx in beams:
            for token, child in node.children.items():
                next_beams.append((score + lm.logprob(ctx, token), uid, child, ctx + [token]))
                uid += 1
        beams = heapq.nlargest(beam_size, next_beams, key=lambda x: x[0])
        if not beams:
            break

    leaves = [
        (score + lm.logprob(ctx, sep), node)
        for score, _, node, ctx in beams
        if node.items
    ]
    leaves.sort(key=lambda x: x[0], reverse=True)

    seen = set(history) if filter_seen else set()
    preds: list[int] = []
    pred_seen: set[int] = set()
    for _, node in leaves:
        for item in node.items:
            if item in seen or item in pred_seen:
                continue
            preds.append(item)
            pred_seen.add(item)
            if len(preds) >= top_k:
                return preds

    for item in popularity_items:
        if item in seen or item in pred_seen:
            continue
        preds.append(item)
        pred_seen.add(item)
        if len(preds) >= top_k:
            break
    return preds


def metric_at_ks(predictions: list[list[int]], targets: list[int], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n = len(targets)
    for k in ks:
        hits = 0.0
        ndcg = 0.0
        mrr = 0.0
        for preds, target in zip(predictions, targets):
            top = preds[:k]
            if target in top:
                rank = top.index(target) + 1
                hits += 1.0
                ndcg += 1.0 / math.log2(rank + 1)
                mrr += 1.0 / rank
        metrics[f"Recall@{k}"] = hits / max(n, 1)
        metrics[f"NDCG@{k}"] = ndcg / max(n, 1)
        metrics[f"MRR@{k}"] = mrr / max(n, 1)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()})
    top_k = max(ks)
    sequences = read_sequences(args.sequences)
    eval_sequences = (
        sequences[: args.max_eval_users] if args.max_eval_users is not None else sequences
    )

    item_tokens, sep, bos, vocab_size = load_item_tokens(
        args.representation,
        args.sid_dir,
        sequences,
    )
    popularity = Counter(item for seq in sequences for item in seq[:-2])
    popularity_items = [item for item, _ in popularity.most_common()]
    popularity_items.extend(item for item in range(len(item_tokens)) if item not in popularity)
    trie = build_trie(item_tokens, popularity)

    train_streams = []
    for seq in sequences:
        train_items = seq[:-1] if args.include_valid_in_train else seq[:-2]
        if train_items:
            train_streams.append(encode_items(train_items, item_tokens, sep, bos))
    print(f"Training {args.representation} {args.order}-gram NTP on {len(train_streams):,} users")
    lm = NgramLM.fit(train_streams, order=args.order, vocab_size=vocab_size, alpha=args.alpha)

    predictions: list[list[int]] = []
    targets: list[int] = []
    rows: list[list[int | str]] = []
    filter_seen = not args.no_filter_seen
    for idx, seq in enumerate(eval_sequences):
        if args.split == "valid":
            history = seq[:-2]
            target = seq[-2]
        else:
            history = seq[:-1]
            target = seq[-1]
        preds = topk_decode(
            lm=lm,
            trie=trie,
            history=history,
            item_tokens=item_tokens,
            sep=sep,
            bos=bos,
            popularity_items=popularity_items,
            top_k=top_k,
            beam_size=args.beam_size,
            max_history_items=args.max_history_items,
            filter_seen=filter_seen,
        )
        predictions.append(preds)
        targets.append(target)
        if idx < args.predictions_head:
            rank = preds.index(target) + 1 if target in preds else 0
            rows.append([idx, target, rank, " ".join(map(str, preds))])

    summary = {
        "representation": args.representation,
        "sid_dir": str(args.sid_dir) if args.representation == "sid" else None,
        "sequences": str(args.sequences),
        "split": args.split,
        "num_train_users": len(train_streams),
        "num_eval_users": len(eval_sequences),
        "num_items": len(item_tokens),
        "token_sequence_length_per_item": len(item_tokens[0]),
        "order": args.order,
        "beam_size": args.beam_size,
        "max_history_items": args.max_history_items,
        "filter_seen": filter_seen,
        "include_valid_in_train": args.include_valid_in_train,
        "vocab_size": vocab_size,
        "metrics": metric_at_ks(predictions, targets, ks),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "predictions_head.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_idx", "target_item", "rank_in_topk", "predicted_items"])
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
