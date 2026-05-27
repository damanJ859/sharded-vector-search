# Implementation Notes

## Purpose

The project provides a reusable local architecture for distributed vector search. It is intended to be adapted to a specific retrieval use case after the corpus, embedding model, and query interface are finalized.

## Core Decisions

- Chroma is used as an optional source for existing embeddings.
- Shards are stored as plain `vectors.npy` plus `metadata.json` files for easy inspection and regeneration.
- FAISS `IndexFlatL2` is used against normalized vectors, then converted back to cosine-style similarity.
- The coordinator queries all local shards and merges results in process.
- Shard assignment uses deterministic LSH projections for reproducible rebuilds.

## Data Privacy

Documents and embeddings can reveal source content. The repository ignores:

- Chroma stores
- raw datasets
- generated shard files
- vector indexes
- local environment files
- logs and caches

Publish only source code, configuration templates, documentation, and synthetic or explicitly approved sample data.

## Reproducibility

To reproduce a shard build, keep these values stable:

- embedding model
- vector dimension
- Chroma collection contents
- `--num-shards`
- `--lsh-bits`
- `--seed`

Changing any of these can alter shard membership or search behavior.

## Limits

- Shards are local files loaded into one process.
- The coordinator searches shards sequentially.
- Text search depends on Chroma exposing a compatible embedding function.
- No authentication, replication, write API, background compaction, or service discovery is included.

These constraints are acceptable for local prototyping and should be addressed before production deployment.
