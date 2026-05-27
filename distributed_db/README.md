# Distributed DB Module

This directory contains the implementation of the local sharded vector search system. The root README explains how to use the project; this README explains how the package is organized and how the implementation works.

## Implementation Overview

The package has four responsibilities:

1. Extract vectors from a source store.
2. Assign vectors to deterministic shards.
3. Build a searchable FAISS index per shard.
4. Query all shards and merge ranked results.

The current source adapter is Chroma. The internal shard format is intentionally simple so another adapter can be added without changing the coordinator or shard node logic.

## Module Map

```text
distributed_db/
├── extract_and_shard.py           # reads source vectors and writes shard folders
├── utils.py                       # LSH projection, shard assignment, normalization
├── run.py                         # CLI commands
├── api.py                         # Flask HTTP API
├── config/config.yaml             # documented defaults
├── coordinator/
│   └── coordinator.py             # loads shards, fans out search, merges results
└── shard_nodes/
    └── shard_node.py              # loads one shard and owns one FAISS index
```

## Data Contract

Every vector record must provide:

- `id`: stable unique document or chunk ID
- `embedding`: one fixed-dimension numeric vector
- `document`: text content or chunk content to retrieve

All embeddings in a shard build must use the same model and vector dimension.

Generated shard folders follow this contract:

```text
distributed_db/data/shards/
├── shard_0/
│   ├── vectors.npy                # float32 matrix: rows are embeddings
│   └── metadata.json              # ids and documents aligned to vector rows
├── shard_1/
│   ├── vectors.npy
│   └── metadata.json
└── shard_info.json                # counts and dimensions per shard
```

`metadata.json` has this shape:

```json
{
  "ids": ["doc-1", "doc-2"],
  "documents": ["first text chunk", "second text chunk"]
}
```

The index position, ID position, and document position must stay aligned.

## Shard Generation Flow

Run:

```bash
python distributed_db/extract_and_shard.py \
  --chroma-path Vec_Store \
  --collection langchain \
  --output-dir distributed_db/data/shards \
  --num-shards 3 \
  --lsh-bits 10 \
  --seed 42
```

What happens internally:

1. `extract_vectors_from_chroma` opens the Chroma collection.
2. It reads embeddings, IDs, and documents.
3. `shard_vectors` calls `assign_to_shard` for each vector.
4. `assign_to_shard` builds a deterministic LSH hash from random projections.
5. The hash integer is mapped with `hash_int % num_shards`.
6. `save_shards` writes one folder per shard.

The same inputs and settings produce the same shard assignment.

## Search Flow

When the package receives a query vector:

1. `Coordinator` loads every `shard_<id>` folder.
2. Each folder becomes one `ShardNode`.
3. Each `ShardNode` loads `vectors.npy` and `metadata.json`.
4. Each `ShardNode` normalizes vectors and builds a FAISS `IndexFlatL2`.
5. `Coordinator.search` sends the query vector to every shard.
6. Each shard returns local nearest neighbors.
7. The coordinator sorts all shard results by score.
8. The top `k` merged results are returned.

The implementation normalizes vectors before FAISS search. L2 distance over normalized vectors is converted into a cosine-style similarity score:

```text
similarity = 1 - (l2_distance / 2)
```

## Python Usage

```python
import numpy as np
from distributed_db.coordinator import Coordinator

coordinator = Coordinator("distributed_db/data/shards")

query_vector = np.random.randn(1536).astype(np.float32)
results = coordinator.search(
    query_vector,
    k=10,
    num_results_per_shard=20,
)

for shard_id, doc_id, score in results:
    _, text = coordinator.get_document(int(shard_id), doc_id)
    print(score, text[:200])
```

For real use, generate `query_vector` with the same embedding model used for the indexed corpus.

## API Usage

Start the API:

```bash
python distributed_db/run.py api
```

Endpoints:

- `GET /health`: service health and database status
- `GET /status`: shard count and vector counts
- `POST /search-vector`: search using a supplied embedding vector
- `POST /search`: search using query text, if Chroma can embed it
- `GET /document/<shard_id>/<doc_id>`: retrieve stored text

Recommended integration path:

1. Your application embeds the user query.
2. Your application calls `/search-vector`.
3. The API returns document IDs, shard IDs, scores, and previews.
4. Your application retrieves full text if needed.
5. Your application passes retrieved context to a UI, reranker, or LLM.

## Configuration

Runtime settings are read from environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DVD_SHARDS_DIR` | shard folder path | `distributed_db/data/shards` |
| `DVD_CHROMA_PATH` | Chroma store path | `Vec_Store` |
| `DVD_CHROMA_COLLECTION` | Chroma collection name | `langchain` |
| `DVD_VECTOR_DIMENSION` | expected query vector dimension | `1536` |
| `DVD_API_HOST` | Flask host | `127.0.0.1` |
| `DVD_API_PORT` | Flask port | `5000` |

Use `.env.example` as the template for local projects. `config/config.yaml` documents equivalent defaults, but the current runtime code reads environment variables and CLI arguments.

## Adapting The Source Adapter

If your embeddings are not in Chroma, keep the output contract the same and replace only the extraction step.

Implement a function that returns:

```python
tuple[np.ndarray, list[str], list[str]]
```

Where:

- the array is shape `(num_documents, embedding_dimension)`
- the first list contains IDs
- the second list contains document text

Then pass those values to `shard_vectors` and `save_shards`.

## Choosing Shard Settings

Start simple:

- `--num-shards 2` or `3` for small local experiments.
- `--num-shards 4` to `8` for larger local corpora.
- Keep `--seed` stable for reproducibility.
- Increase `--lsh-bits` only when you are intentionally changing the partitioning behavior.

After generating shards, check:

- Are vectors distributed reasonably evenly?
- Does each shard fit in memory?
- Is query latency acceptable?
- Do top results still make sense for known queries?

## Package Commands

From the repository root:

```bash
bash START_HERE.sh status
bash START_HERE.sh examples
bash START_HERE.sh api
bash START_HERE.sh shell
```

After editable installation:

```bash
sharded-vector-search status
sharded-vector-search examples
sharded-vector-search api
sharded-vector-search shell
```

## Safe Repository Practices

Do not commit generated shard files or private data. These are ignored because they may contain recoverable source content:

- `distributed_db/data/`
- Chroma stores
- raw datasets
- `.env`
- logs
- caches
- vector index files

Use synthetic data or explicitly approved sample data if the repository needs examples.

## Extension Points

Common next changes:

- Add a source adapter for CSV, Parquet, PostgreSQL, S3, or another vector store.
- Replace local sequential fan-out with parallel shard queries.
- Move shards behind network services.
- Add metadata filtering.
- Add reranking after top-k retrieval.
- Add persistent FAISS indexes instead of building them on startup.
- Add tests around shard assignment, vector dimensions, and API validation.

Keep the public contract stable: query vector in, ranked document references out.
