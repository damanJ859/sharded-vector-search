# Quick Start

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Generate Shards

Create shards from a local Chroma store:

```bash
python distributed_db/extract_and_shard.py \
  --chroma-path Vec_Store \
  --collection langchain \
  --output-dir distributed_db/data/shards \
  --num-shards 3
```

`distributed_db/data/` is generated output and is ignored by Git.

## Run Locally

```bash
bash START_HERE.sh status
bash START_HERE.sh examples
bash START_HERE.sh api
bash START_HERE.sh shell
```

The API uses:

- `DVD_SHARDS_DIR`, default `distributed_db/data/shards`
- `DVD_CHROMA_PATH`, default `Vec_Store`
- `DVD_CHROMA_COLLECTION`, default `langchain`
- `DVD_VECTOR_DIMENSION`, default `1536`
- `DVD_API_HOST`, default `127.0.0.1`
- `DVD_API_PORT`, default `5000`

## API Checks

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

The vector length must match `DVD_VECTOR_DIMENSION`.

Text search requires a Chroma collection that can embed the query text:

```bash
curl -X POST http://127.0.0.1:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "sample query", "k": 5}'
```

## Public Repository Checklist

- Keep `.env` local.
- Keep `Vec_Store/`, `Pre_stored_Vec_Store/`, `rag_dataset/`, and `distributed_db/data/` out of Git.
- Commit source code, docs, `.env.example`, `.gitignore`, `requirements.txt`, and license files.
