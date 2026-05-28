from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer NTP over Beauty SID tokens.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/processed"),
    )
    parser.add_argument(
        "--sid-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/sid_rqopq"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/amazon_beauty_v2/transformer_ntp"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-history-items", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--eval-users", type=int, default=2000)
    parser.add_argument("--beam-size", type=int, default=128)
    parser.add_argument("--ks", type=str, default="5,10,20")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def read_jsonl_samples(path: Path, limit: int | None) -> list[dict[str, int | list[int]]]:
    samples: list[dict[str, int | list[int]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            samples.append(
                {
                    "history": [int(x) for x in rec["history"]],
                    "target": int(rec["target"]),
                }
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_sid_tokens(sid_dir: Path) -> tuple[list[list[int]], int, int, int]:
    codes = np.load(sid_dir / "sid_codes_offset.npy", mmap_mode="r")
    item_ids = np.load(sid_dir / "itemid.npy", mmap_mode="r")
    if codes.shape[0] != item_ids.shape[0]:
        raise ValueError("sid_codes_offset.npy and itemid.npy row counts differ")
    if not np.array_equal(np.asarray(item_ids, dtype=np.int64), np.arange(codes.shape[0])):
        raise ValueError("itemid.npy must be contiguous 0..N-1")
    item_tokens = [np.asarray(codes[i], dtype=np.int64).astype(int).tolist() for i in range(codes.shape[0])]
    max_sid_token = int(np.max(codes))
    sep = max_sid_token + 1
    bos = max_sid_token + 2
    vocab_size = bos + 1
    return item_tokens, sep, bos, vocab_size


def encode_history(history: list[int], item_tokens: list[list[int]], sep: int, bos: int) -> list[int]:
    tokens = [bos]
    for item in history:
        tokens.extend(item_tokens[item])
        tokens.append(sep)
    return tokens


class SidNtpDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, int | list[int]]],
        item_tokens: list[list[int]],
        sep: int,
        bos: int,
        max_history_items: int,
    ) -> None:
        self.samples = samples
        self.item_tokens = item_tokens
        self.sep = sep
        self.bos = bos
        self.max_history_items = max_history_items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        sample = self.samples[idx]
        history = list(sample["history"])  # type: ignore[arg-type]
        target = int(sample["target"])
        if self.max_history_items > 0:
            history = history[-self.max_history_items :]
        context = encode_history(history, self.item_tokens, self.sep, self.bos)
        target_tokens = self.item_tokens[target]
        full = context + target_tokens
        labels = [IGNORE_INDEX] * (len(context) - 1) + target_tokens
        return full[:-1], labels


def collate_batch(batch: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (x, y) in enumerate(batch):
        input_ids[row, : len(x)] = torch.tensor(x, dtype=torch.long)
        labels[row, : len(y)] = torch.tensor(y, dtype=torch.long)
    return input_ids, labels


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        max_len: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = input_ids.eq(self.pad_id)
        x = self.blocks(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.lm_head(self.norm(x))


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

    def sort_items(node: TrieNode) -> None:
        if node.items:
            node.items.sort(key=lambda item: (-popularity[item], item))
        for child in node.children.values():
            sort_items(child)

    sort_items(root)
    return root


@torch.no_grad()
def constrained_decode(
    model: DecoderOnlyTransformer,
    trie: TrieNode,
    history: list[int],
    item_tokens: list[list[int]],
    sep: int,
    bos: int,
    popularity_items: list[int],
    top_k: int,
    beam_size: int,
    max_history_items: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    hist = history[-max_history_items:] if max_history_items > 0 else history
    context = encode_history(hist, item_tokens, sep, bos)
    beams: list[tuple[float, int, TrieNode, list[int]]] = [(0.0, 0, trie, context)]
    uid = 1
    for _ in range(len(item_tokens[0])):
        active = [(score, node, tokens) for score, _, node, tokens in beams if node.children]
        if not active:
            break
        max_len = max(len(tokens) for _, _, tokens in active)
        input_ids = torch.full(
            (len(active), max_len),
            model.pad_id,
            dtype=torch.long,
            device=device,
        )
        last_positions = []
        for row, (_, _, tokens) in enumerate(active):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            last_positions.append(len(tokens) - 1)
        logits = model(input_ids)

        next_beams: list[tuple[float, int, TrieNode, list[int]]] = []
        for row, (score, node, tokens) in enumerate(active):
            row_logits = logits[row, last_positions[row]]
            allowed_tokens = list(node.children.keys())
            allowed = torch.tensor(allowed_tokens, dtype=torch.long, device=device)
            log_probs = torch.log_softmax(row_logits[allowed], dim=-1)
            keep = min(len(allowed_tokens), beam_size)
            values, indices = torch.topk(log_probs, k=keep)
            for value, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                token = allowed_tokens[index]
                next_beams.append((score + float(value), uid, node.children[token], tokens + [token]))
                uid += 1
        beams = sorted(next_beams, key=lambda x: x[0], reverse=True)[:beam_size]
        if not beams:
            break

    leaves = [(score, node) for score, _, node, _ in beams if node.items]
    leaves.sort(key=lambda x: x[0], reverse=True)
    seen = set(history)
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


def metrics_at_k(predictions: list[list[int]], targets: list[int], ks: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    n = len(targets)
    for k in ks:
        recall = 0.0
        ndcg = 0.0
        mrr = 0.0
        for preds, target in zip(predictions, targets):
            top = preds[:k]
            if target in top:
                rank = top.index(target) + 1
                recall += 1.0
                ndcg += 1.0 / math.log2(rank + 1)
                mrr += 1.0 / rank
        out[f"Recall@{k}"] = recall / max(n, 1)
        out[f"NDCG@{k}"] = ndcg / max(n, 1)
        out[f"MRR@{k}"] = mrr / max(n, 1)
    return out


def samples_from_sequences(sequences: list[list[int]], split: str) -> Iterable[tuple[list[int], int]]:
    for seq in sequences:
        if split == "valid":
            yield seq[:-2], seq[-2]
        else:
            yield seq[:-1], seq[-1]


def evaluate(
    model: DecoderOnlyTransformer,
    sequences: list[list[int]],
    split: str,
    item_tokens: list[list[int]],
    sep: int,
    bos: int,
    trie: TrieNode,
    popularity_items: list[int],
    ks: list[int],
    beam_size: int,
    max_history_items: int,
    device: torch.device,
    eval_users: int | None,
) -> dict[str, float]:
    predictions: list[list[int]] = []
    targets: list[int] = []
    pairs = list(samples_from_sequences(sequences, split))
    if eval_users is not None:
        pairs = pairs[:eval_users]
    for history, target in pairs:
        predictions.append(
            constrained_decode(
                model=model,
                trie=trie,
                history=history,
                item_tokens=item_tokens,
                sep=sep,
                bos=bos,
                popularity_items=popularity_items,
                top_k=max(ks),
                beam_size=beam_size,
                max_history_items=max_history_items,
                device=device,
            )
        )
        targets.append(target)
    return metrics_at_k(predictions, targets, ks)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    ks = sorted({int(k.strip()) for k in args.ks.split(",") if k.strip()})
    item_tokens, sep, bos, vocab_size = load_sid_tokens(args.sid_dir)
    pad_id = vocab_size
    model_vocab_size = vocab_size + 1
    token_len = len(item_tokens[0])
    max_len = 1 + args.max_history_items * (token_len + 1) + token_len

    train_samples = read_jsonl_samples(
        args.processed_dir / "splits" / "train.jsonl",
        limit=args.limit_train_samples,
    )
    sequences = read_sequences(args.processed_dir / "sequences.jsonl")
    popularity = Counter(item for seq in sequences for item in seq[:-2])
    popularity_items = [item for item, _ in popularity.most_common()]
    popularity_items.extend(item for item in range(len(item_tokens)) if item not in popularity)
    trie = build_trie(item_tokens, popularity)

    dataset = SidNtpDataset(
        samples=train_samples,
        item_tokens=item_tokens,
        sep=sep,
        bos=bos,
        max_history_items=args.max_history_items,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_batch(batch, pad_id),
    )
    model = DecoderOnlyTransformer(
        vocab_size=model_vocab_size,
        pad_id=pad_id,
        max_len=max_len,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    history_rows: list[dict[str, float | int]] = []
    best_valid = -1.0
    start_time = time.time()
    print(
        f"Training Transformer NTP on {len(dataset):,} samples, "
        f"vocab={model_vocab_size}, sid_len={token_len}, device={device}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        token_count = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            supervised = labels.ne(IGNORE_INDEX).sum().item()
            loss_sum += float(loss.item()) * supervised
            token_count += int(supervised)
        train_loss = loss_sum / max(token_count, 1)
        row: dict[str, float | int] = {"epoch": epoch, "train_loss": train_loss}
        print(f"epoch={epoch} train_loss={train_loss:.5f}", flush=True)
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            valid_metrics = evaluate(
                model=model,
                sequences=sequences,
                split="valid",
                item_tokens=item_tokens,
                sep=sep,
                bos=bos,
                trie=trie,
                popularity_items=popularity_items,
                ks=ks,
                beam_size=args.beam_size,
                max_history_items=args.max_history_items,
                device=device,
                eval_users=args.eval_users,
            )
            row.update({f"valid_{k}": v for k, v in valid_metrics.items()})
            print(f"  valid {valid_metrics}", flush=True)
            valid_score = valid_metrics.get("Recall@10", next(iter(valid_metrics.values())))
            if valid_score > best_valid:
                best_valid = valid_score
                if args.save_model:
                    torch.save(model.state_dict(), args.output_dir / "best_model.pt")
        history_rows.append(row)

    test_metrics = evaluate(
        model=model,
        sequences=sequences,
        split="test",
        item_tokens=item_tokens,
        sep=sep,
        bos=bos,
        trie=trie,
        popularity_items=popularity_items,
        ks=ks,
        beam_size=args.beam_size,
        max_history_items=args.max_history_items,
        device=device,
        eval_users=args.eval_users,
    )
    summary = {
        "processed_dir": str(args.processed_dir),
        "sid_dir": str(args.sid_dir),
        "num_train_samples": len(dataset),
        "num_items": len(item_tokens),
        "sid_length": token_len,
        "vocab_size_without_pad": vocab_size,
        "model_vocab_size": model_vocab_size,
        "pad_id": pad_id,
        "device": str(device),
        "config": vars(args),
        "history": history_rows,
        "test_metrics": test_metrics,
        "elapsed_seconds": time.time() - start_time,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    with (args.output_dir / "train_history.csv").open("w", newline="", encoding="utf-8") as f:
        if history_rows:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
    if args.save_model:
        torch.save(model.state_dict(), args.output_dir / "last_model.pt")
    print(json.dumps({"test_metrics": test_metrics, "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
