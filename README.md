# Distributed Vector Database

A Python reference implementation for sharding embedding vectors across local FAISS-backed nodes and querying them through a coordinator or REST API.

The project is designed for retrieval systems where a single vector store needs to be split into smaller searchable partitions. The current pipeline can extract vectors from a Chroma collection, assign them to deterministic LSH-based shards, build a FAISS index per shard, and merge top-k results across shards.

## Use Cases

- Prototype retrieval infrastructure before adding a full RAG search layer.
- Split an existing embedding corpus into independently searchable partitions.
- Compare shard distributions and query latency for different embedding datasets.
- Reuse the coordinator/shard layout with domain-specific document collections.

## What Is Included

- Chroma extraction and sharding pipeline.
- Deterministic LSH shard assignment.
- FAISS-backed shard nodes using normalized vector search.
- Coordinator that queries all shards and merges ranked results.
- Flask API for status, vector search, optional text search, and document retrieval.
- CLI entry point for status checks, examples, interactive shell, and API startup.

## What Is Not Committed

The repository intentionally excludes local datasets, generated shard files, vector indexes, Chroma stores, logs, virtual environments, and secrets. See `.gitignore`.

Expected local-only directories include:

- `Vec_Store/`
- `Pre_stored_Vec_Store/`
- `rag_dataset/`
- `distributed_db/data/`
- `.venv/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For editable package installation:

```bash
pip install -e .
```

Edit `.env` if your Chroma store, shard output path, API host, or vector dimension differs.

## Build Shards

```bash
python distributed_db/extract_and_shard.py \
  --chroma-path Vec_Store \
  --collection langchain \
  --output-dir distributed_db/data/shards \
  --num-shards 3
```

The output directory contains one `shard_<id>/` folder per shard with `vectors.npy` and `metadata.json`, plus a `shard_info.json` summary.

## Run

```bash
bash START_HERE.sh status
bash START_HERE.sh examples
bash START_HERE.sh api
bash START_HERE.sh shell
```

The API defaults to `http://127.0.0.1:5000`.

## API

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/status
```

Vector search:

```bash
curl -X POST http://127.0.0.1:5000/search-vector \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10}'
```

Text search is available when `DVD_CHROMA_PATH` points to a Chroma collection that can embed query text:

```bash
curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "reconstructive surgery", "k": 5}'
```

## Architecture

```text
Chroma collection or embedding source
        |
        v
extract_and_shard.py
        |
        v
Deterministic LSH assignment
        |
        v
shard_0/       shard_1/       shard_n/
FAISS index    FAISS index    FAISS index
        \          |          /
         \         |         /
          v        v        v
             Coordinator
                 |
                 v
              Top-k results
```

## Adapting to Another Dataset

1. Generate embeddings with a consistent model and dimension.
2. Store them in Chroma or adapt `extract_and_shard.py` to read your source.
3. Set `DVD_VECTOR_DIMENSION` to the embedding dimension.
4. Rebuild shards with the desired `--num-shards`.
5. Start the coordinator or API against the generated shard directory.

## Notes

This implementation is a local reference system. For production deployment, add authentication, request limits, structured logging, persistent service supervision, backup policies, and either distributed workers or networked shard services.
