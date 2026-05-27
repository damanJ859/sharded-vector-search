"""REST API for Distributed Vector Database."""

import os
from pathlib import Path

from flask import Flask, request, jsonify
import numpy as np
from coordinator.coordinator import Coordinator

app = Flask(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
SHARDS_DIR = os.getenv("DVD_SHARDS_DIR", str(Path(__file__).resolve().parent / "data" / "shards"))
CHROMA_PATH = os.getenv("DVD_CHROMA_PATH", str(ROOT_DIR / "Vec_Store"))
CHROMA_COLLECTION = os.getenv("DVD_CHROMA_COLLECTION", "langchain")
VECTOR_DIMENSION = int(os.getenv("DVD_VECTOR_DIMENSION", "1536"))
API_HOST = os.getenv("DVD_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("DVD_API_PORT", "5000"))

coordinator = None
chroma_client = None
chroma_collection = None


def init_system():
    """Initialize distributed database system."""
    global coordinator, chroma_client, chroma_collection
    if coordinator is not None:
        return

    print("Initializing distributed vector database...")
    coordinator = Coordinator(SHARDS_DIR)
    if os.path.isdir(CHROMA_PATH):
        import chromadb

        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = chroma_client.get_collection(CHROMA_COLLECTION)
    print("System initialized")


@app.before_request
def ensure_initialized():
    init_system()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = coordinator.get_status()
    return jsonify({
        "status": "healthy",
        "database": status
    }), 200


@app.route('/search', methods=['POST'])
def search():
    """
    Search for similar documents.
    
    Expected JSON:
    {
        "query_text": "string to search",
        "k": 10,
        "num_results_per_shard": 20
    }
    """
    data = request.get_json()
    
    if not data or 'query_text' not in data:
        return jsonify({"error": "Missing 'query_text' field"}), 400
    
    query_text = data['query_text']
    k = data.get('k', 10)
    num_results_per_shard = data.get('num_results_per_shard', 20)
    
    try:
        if chroma_collection is None:
            return jsonify({"error": "Text search requires DVD_CHROMA_PATH to point to a Chroma store"}), 503
        
        embedding_function = getattr(chroma_collection, "_embedding_function", None)
        if embedding_function is None:
            return jsonify({"error": "Chroma collection does not expose a query embedding function"}), 503

        query_embedding = np.array(embedding_function([query_text])[0], dtype=np.float32)
        if query_embedding.size == 0:
            return jsonify({"error": "Could not generate embedding for query"}), 400
        
        results = coordinator.search(query_embedding, k=k, num_results_per_shard=num_results_per_shard)
        
        formatted_results = []
        for shard_id, doc_id, score in results:
            shard_id_int = int(shard_id)
            _, document = coordinator.get_document(shard_id_int, doc_id)
            
            formatted_results.append({
                "shard_id": shard_id,
                "document_id": doc_id,
                "similarity_score": float(score),
                "document_preview": document[:200] + "..." if document and len(document) > 200 else document
            })
        
        return jsonify({
            "query": query_text,
            "num_results": len(formatted_results),
            "results": formatted_results
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search-vector', methods=['POST'])
def search_vector():
    """
    Search using a raw embedding vector.
    
    Expected JSON:
    {
        "vector": [list of floats],
        "k": 10
    }
    """
    data = request.get_json()
    
    if not data or 'vector' not in data:
        return jsonify({"error": "Missing 'vector' field"}), 400
    
    vector = np.array(data['vector'], dtype=np.float32)
    k = data.get('k', 10)

    if vector.ndim != 1 or vector.size != VECTOR_DIMENSION:
        return jsonify({"error": f"Expected a {VECTOR_DIMENSION}-dimensional vector"}), 400
    
    try:
        results = coordinator.search(vector, k=k)
        
        formatted_results = []
        for shard_id, doc_id, score in results:
            shard_id_int = int(shard_id)
            _, document = coordinator.get_document(shard_id_int, doc_id)
            
            formatted_results.append({
                "shard_id": shard_id,
                "document_id": doc_id,
                "similarity_score": float(score),
                "document_preview": document[:200] + "..." if document and len(document) > 200 else document
            })
        
        return jsonify({
            "num_results": len(formatted_results),
            "results": formatted_results
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Get system status."""
    status = coordinator.get_status()
    return jsonify(status), 200


@app.route('/document/<shard_id>/<doc_id>', methods=['GET'])
def get_document(shard_id, doc_id):
    """Retrieve a specific document."""
    try:
        shard_id_int = int(shard_id)
        retrieved_id, document = coordinator.get_document(shard_id_int, doc_id)
        
        if document is None:
            return jsonify({"error": "Document not found"}), 404
        
        return jsonify({
            "shard_id": shard_id,
            "document_id": retrieved_id,
            "document": document
        }), 200
    
    except ValueError:
        return jsonify({"error": "Invalid shard_id"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_system()
    app.run(debug=False, host=API_HOST, port=API_PORT)
