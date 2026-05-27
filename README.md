# Sharded Vector Search

Sharded Vector Search is a local reference implementation for splitting an embedding corpus into multiple searchable shards, building a FAISS index for each shard, and querying all shards through one coordinator.

Use this project when you already have, or plan to create, document embeddings and want a clear starting point for scalable retrieval infrastructure before adding a full RAG or search application on top.

## What Problem This Solves

Most retrieval systems start with one vector store. That is simple, but it becomes harder to reason about as the corpus grows, experiments multiply, or data needs to be partitioned by tenant, domain, time period, region, or storage boundary.

This project shows one practical pattern:

```text
documents
  -> embeddings
  -> deterministic shard assignment
  -> one FAISS index per shard
  -> coordinator queries every shard
  -> merged top-k results
```

The implementation is intentionally small and inspectable. It is a package foundation, not a managed database service.

## Good Use Cases

- RAG retrieval layer prototypes.
- Semantic search over internal documents.
- Search experiments across different embedding models.
- Partitioned retrieval for multiple tenants or collections.
- Local benchmarking of shard counts and retrieval latency.
- Educational reference for vector sharding, FAISS search, and result merging.
- A starting point before moving shards into separate services or containers.

## Not A Good Fit For

- A drop-in replacement for a managed vector database.
- High-availability production search without additional infrastructure.
- Systems needing authentication, replication, writes, background indexing, or networked shard services out of the box.
- Public repositories containing private embeddings or source documents.

## Repository Contents

```text
.
├── distributed_db/
│   ├── api.py                         # Flask API
│   ├── extract_and_shard.py           # Chroma extraction and shard generation
│   ├── run.py                         # CLI entry point
│   ├── utils.py                       # LSH, normalization, JSON helpers
│   ├── config/config.yaml             # documented configuration defaults
│   ├── coordinator/coordinator.py     # query fan-out and result merge
│   └── shard_nodes/shard_node.py      # local FAISS shard implementation
├── .env.example                       # local environment template
├── pyproject.toml                     # installable package metadata
├── requirements.txt                   # dependency list
├── QUICK_START.md                     # compact command reference
└── START_HERE.sh                      # convenience wrapper
```

Generated data is not committed. Local vector stores, raw datasets, shard files, logs, caches, virtual environments, and secrets are ignored by `.gitignore`.

## How The System Works

1. You create embeddings for your documents.
2. The embeddings are stored in Chroma, or you adapt the extraction step to another source.
3. `extract_and_shard.py` reads vectors, IDs, and document text.
4. Each vector is assigned to a shard using deterministic LSH projection.
5. Each shard is saved as:
   - `vectors.npy`
   - `metadata.json`
6. A `ShardNode` loads one shard and builds a FAISS index.
7. The `Coordinator` loads every shard, sends each query vector to all shards, merges scored results, and returns the top `k`.
8. You use the coordinator directly from Python or expose it through the Flask API.

## Setup

Clone the repo and create a local environment:

```bash
git clone https://github.com/damanJ859/sharded-vector-search.git
cd sharded-vector-search

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

Edit `.env` for your local paths and embedding dimension:

```bash
DVD_SHARDS_DIR=distributed_db/data/shards
DVD_CHROMA_PATH=Vec_Store
DVD_CHROMA_COLLECTION=langchain
DVD_VECTOR_DIMENSION=1536
DVD_API_HOST=127.0.0.1
DVD_API_PORT=5000
```

## Step 1: Prepare Your Data

You need a collection of documents and embeddings.

Recommended starting path:

1. Chunk your source documents.
2. Generate one embedding per chunk using one embedding model.
3. Store each chunk with:
   - unique ID
   - embedding vector
   - document text or chunk text
4. Persist the result in a Chroma collection.

All vectors in one run must have the same dimension. For example, OpenAI `text-embedding-3-small` and many other embedding models produce fixed-size vectors. Set `DVD_VECTOR_DIMENSION` to match your model.

## Step 2: Build Shards

If your embeddings are in Chroma:

```bash
python distributed_db/extract_and_shard.py \
  --chroma-path Vec_Store \
  --collection langchain \
  --output-dir distributed_db/data/shards \
  --num-shards 3 \
  --lsh-bits 10 \
  --seed 42
```

This creates:

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

Choose `--num-shards` based on your corpus size and experiment goals. For a small local corpus, 2-4 shards is enough. For larger experiments, increase it and compare latency, memory use, and shard balance.

## Step 3: Check The Shards

```bash
bash START_HERE.sh status
```

or:

```bash
python distributed_db/run.py status
```

This loads all shard folders and reports shard count, vector count, and dimensions.

## Step 4: Query From Python

```python
import numpy as np
from distributed_db.coordinator import Coordinator

coordinator = Coordinator("distributed_db/data/shards")

query_vector = np.random.randn(1536).astype(np.float32)
results = coordinator.search(query_vector, k=5)

for shard_id, document_id, score in results:
    _, document = coordinator.get_document(int(shard_id), document_id)
    print(score, document_id, document[:200])
```

In a real application, replace the random vector with an embedding generated by the same model used to build the corpus.

## Step 5: Run The API

```bash
bash START_HERE.sh api
```

Health and status:

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

The request vector must contain exactly `DVD_VECTOR_DIMENSION` floats.

Text search:

```bash
curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "sample search query", "k": 5}'
```

Text search only works when your configured Chroma collection can embed query text. If you use a different embedding provider, generate the query embedding in your application and call `/search-vector`.

## Adapting This To Any Use Case

For a new domain, keep the retrieval contract stable:

1. Choose a document chunking strategy.
2. Choose one embedding model.
3. Generate embeddings for every chunk.
4. Store vectors, IDs, and text in Chroma or adapt `extract_and_shard.py`.
5. Build shards.
6. Use the coordinator or API for retrieval.
7. Send retrieved text to your search UI, ranking layer, or LLM.

Examples:

- Legal documents: chunk by section or clause, retrieve relevant clauses.
- Medical literature: chunk abstracts or full text, retrieve supporting passages.
- Product support: chunk help articles, retrieve troubleshooting steps.
- Company knowledge base: chunk policies, tickets, and docs, retrieve internal answers.
- Code search: embed functions or files, retrieve likely relevant source snippets.

## Reproducibility Rules

To reproduce a shard build, keep these fixed:

- source documents
- chunking strategy
- embedding model
- vector dimension
- Chroma collection contents
- `--num-shards`
- `--lsh-bits`
- `--seed`

Changing any of these can change shard assignment or search results.

## Privacy Rules

Do not commit generated or sensitive data:

- `.env`
- raw datasets
- Chroma stores
- `distributed_db/data/`
- `vectors.npy`
- document metadata generated from private text
- logs and caches

Commit only source code, docs, dependency files, config templates, and approved synthetic sample data.

## Production Path

Before production use, add:

- authentication and authorization
- request size limits
- rate limiting
- structured logs
- tests for shard generation and search behavior
- monitoring and error tracking
- Docker or deployment manifests
- backup and rebuild procedures
- networked shard services if shards must run on separate machines

The current repository is a clean implementation base for those additions.
