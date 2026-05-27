# Distributed DB Module

This module contains the local distributed vector search implementation.

## Components

- `extract_and_shard.py`: extracts embeddings from Chroma and writes deterministic shard files.
- `utils.py`: LSH assignment, vector normalization, and JSON helpers.
- `shard_nodes/shard_node.py`: loads a shard from disk and builds a FAISS index.
- `coordinator/coordinator.py`: queries all shards and merges top-k results.
- `api.py`: Flask API for health, status, vector search, optional text search, and document retrieval.
- `run.py`: CLI wrapper for status, examples, API startup, and shell mode.
- `config/config.yaml`: documented defaults for local configuration.

## Data Layout

Generated shard files use this layout:

```text
distributed_db/data/shards/
├── shard_0/
│   ├── vectors.npy
│   └── metadata.json
├── shard_1/
│   ├── vectors.npy
│   └── metadata.json
└── shard_info.json
```

The `data/` directory is ignored because it can contain private documents, embeddings, and generated indexes.

## Query Flow

1. A query vector is normalized.
2. Each shard searches its FAISS index for local nearest neighbors.
3. The coordinator combines shard results.
4. Results are sorted by similarity score and truncated to `k`.

## Sharding Logic

`assign_to_shard` uses deterministic random projections:

1. Project each vector into `lsh_bits` dimensions.
2. Convert projection signs into a binary hash.
3. Convert the hash to an integer.
4. Assign `hash_int % num_shards`.

This keeps assignment reproducible across runs when `num_shards`, `lsh_bits`, and `seed` stay fixed.

## Local Commands

```bash
python run.py status
python run.py examples
python run.py api
python run.py shell
```

From the repository root, prefer:

```bash
bash START_HERE.sh status
```

## Configuration

Runtime configuration is read from environment variables:

- `DVD_SHARDS_DIR`
- `DVD_CHROMA_PATH`
- `DVD_CHROMA_COLLECTION`
- `DVD_VECTOR_DIMENSION`
- `DVD_API_HOST`
- `DVD_API_PORT`

Use `.env.example` as a template for local configuration.

## Production Considerations

This module is a reference implementation. Before serving sensitive or high-volume traffic, add authentication, rate limiting, structured logs, error monitoring, input size limits, deployment manifests, and tests around the embedding source used for text search.
