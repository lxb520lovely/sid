# Amazon Beauty SID Project

This folder is the Beauty-specific pipeline. It uses the existing RQ/OPQ
quantization code as a method reference, but does not force Amazon Beauty into
the older `title_emb.npy + image_emb.npy + item_feat.npz` multimodal interface.

## Design

Open-source TIGER/RQVAE pipelines usually do three things for Amazon 2014:

- use the 5-core Amazon reviews,
- build chronological user sequences and leave-one-out valid/test splits,
- build item embeddings from item metadata text, then quantize those embeddings
  into semantic IDs.

This pipeline follows that shape:

```text
reviews/meta
  -> dataset.py
     items.csv, item_text.csv, sequences.jsonl, splits/*.jsonl
  -> embed_items.py
     item_embeddings.npy
  -> build_rqopq_sid.py
     sid_codes.npy, sid_codes_offset.npy, metrics.json
  -> ngram_ntp.py
     quick next-token generation smoke test
```

## 1. Prepare Dataset

```bash
.venv/bin/python -m amazon_beauty_sid.dataset \
  --reviews data/amazon_beauty/raw/reviews_Beauty_5.json.gz \
  --metadata data/amazon_beauty/raw/meta_Beauty.json.gz \
  --output-dir data/amazon_beauty_v2/processed \
  --min-user-items 5 \
  --item-order first_seen
```

Important outputs:

```text
data/amazon_beauty_v2/processed/items.csv
data/amazon_beauty_v2/processed/item_text.csv
data/amazon_beauty_v2/processed/sequences.jsonl
data/amazon_beauty_v2/processed/splits/train.jsonl
data/amazon_beauty_v2/processed/splits/valid.jsonl
data/amazon_beauty_v2/processed/splits/test.jsonl
```

`item_text.csv` uses metadata text in this form:

```text
Title: ...; Brand: ...; Categories: ...; Price: ...; SalesRank: ...
```

Review text is intentionally not used for item SID construction.

## 2. Embed Items

Dependency-light local embedding:

```bash
.venv/bin/python -m amazon_beauty_sid.embed_items \
  --item-text data/amazon_beauty_v2/processed/item_text.csv \
  --output-dir data/amazon_beauty_v2/processed \
  --encoder hashing-svd \
  --emb-dim 256
```

Closer to TIGER/RQVAE-style experiments, if `sentence-transformers` is
installed:

```bash
.venv/bin/python -m amazon_beauty_sid.embed_items \
  --item-text data/amazon_beauty_v2/processed/item_text.csv \
  --output-dir data/amazon_beauty_v2/processed \
  --encoder sentence-transformers \
  --model-name sentence-transformers/sentence-t5-base
```

## 3. Build RQ-OPQ SID

```bash
.venv/bin/python -m amazon_beauty_sid.build_rqopq_sid \
  --item-embeddings data/amazon_beauty_v2/processed/item_embeddings.npy \
  --itemid data/amazon_beauty_v2/processed/itemid.npy \
  --output-dir data/amazon_beauty_v2/sid_rqopq \
  --code-dim 32 \
  --rq-clusters 256 \
  --rq-levels 3 \
  --opq-subspaces 2 \
  --opq-clusters 128
```

The SID builder saves:

```text
sid_codes_quantized.npy   raw RQ/OPQ tokens
sid_codes.npy             final 5-token SID
sid_codes_offset.npy      position-offset tokens for generative models
sid_codebook_sizes.npy    codebook size per SID position
metrics.json
```

Collision suffixes are disabled by default. If several items share the same
5-token SID, downstream constrained decoding expands that SID to its item group,
so collision ambiguity is reflected in Recall/NDCG.

## 4. Quick NTP Smoke Test

SID representation:

```bash
.venv/bin/python -m amazon_beauty_sid.ngram_ntp \
  --sequences data/amazon_beauty_v2/processed/sequences.jsonl \
  --sid-dir data/amazon_beauty_v2/sid_rqopq \
  --representation sid \
  --output-dir data/amazon_beauty_v2/ntp_sid_ngram \
  --split test \
  --ks 5,10,20
```

Atomic item-token baseline:

```bash
.venv/bin/python -m amazon_beauty_sid.ngram_ntp \
  --sequences data/amazon_beauty_v2/processed/sequences.jsonl \
  --representation item \
  --output-dir data/amazon_beauty_v2/ntp_item_ngram \
  --split test \
  --ks 5,10,20
```

This n-gram evaluator is only a smoke test. For the Transformer version, use
`transformer_ntp.py`.

```bash
.venv/bin/python -m amazon_beauty_sid.transformer_ntp \
  --processed-dir data/amazon_beauty_v2/processed \
  --sid-dir data/amazon_beauty_v2/sid_rqopq \
  --output-dir data/amazon_beauty_v2/transformer_ntp \
  --epochs 20 \
  --batch-size 128 \
  --max-history-items 50
```
