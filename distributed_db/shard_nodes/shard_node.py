"""Shard Node: Stores and indexes vectors locally using FAISS."""

import json
import os

import faiss
import numpy as np

from utils import normalize_vector


class ShardNode:
    """Individual shard node for storing and searching vectors."""
    
    def __init__(self, shard_id: int, data_dir: str):
        """
        Initialize shard node.
        
        Args:
            shard_id: ID of this shard
            data_dir: Directory containing shard data
        """
        self.shard_id = shard_id
        self.data_dir = data_dir
        self.index = None
        self.vectors = None
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadata: dict = {}
        
        self.load_data()
        self.build_index()
    
    def load_data(self) -> None:
        """Load vectors and metadata from disk."""
        vectors_path = os.path.join(self.data_dir, "vectors.npy")
        metadata_path = os.path.join(self.data_dir, "metadata.json")

        if not os.path.exists(vectors_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Shard {self.shard_id} is missing vectors.npy or metadata.json")
        
        self.vectors = np.load(vectors_path).astype(np.float32)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            self.ids = metadata["ids"]
            self.documents = metadata["documents"]

        if len(self.vectors) != len(self.ids) or len(self.ids) != len(self.documents):
            raise ValueError(f"Shard {self.shard_id} has inconsistent vector and metadata counts")
        
        print(f"Shard {self.shard_id}: Loaded {len(self.vectors)} vectors")
    
    def build_index(self) -> None:
        """Build FAISS index for fast similarity search."""
        dimension = self.vectors.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        
        # Normalize vectors for cosine similarity
        normalized = np.zeros_like(self.vectors)
        for i, vec in enumerate(self.vectors):
            normalized[i] = normalize_vector(vec)
        
        self.index.add(normalized)
        print(f"Shard {self.shard_id}: Built FAISS index for {len(self.vectors)} vectors")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """
        Search for similar vectors in this shard.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        query_normalized = normalize_vector(query_vector).reshape(1, -1).astype(np.float32)
        
        distances, indices = self.index.search(query_normalized, min(k, len(self.vectors)))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # Valid result
                # Convert L2 distance back to cosine similarity
                # For normalized vectors: cosine_sim = 1 - (L2_dist / 2)
                similarity = 1.0 - (distance / 2.0)
                results.append((self.ids[idx], float(similarity)))
        
        return results
    
    def get_info(self) -> dict:
        """Get shard information."""
        return {
            "shard_id": self.shard_id,
            "vector_count": len(self.vectors),
            "dimension": self.vectors.shape[1] if len(self.vectors) > 0 else 0
        }


def load_all_shards(shards_dir: str) -> dict[int, ShardNode]:
    """Load all shard nodes from disk."""
    shards = {}
    if not os.path.isdir(shards_dir):
        return shards
    
    for item in sorted(os.listdir(shards_dir)):
        if item.startswith("shard_") and os.path.isdir(os.path.join(shards_dir, item)):
            try:
                shard_id = int(item.split("_")[1])
                shard_path = os.path.join(shards_dir, item)
                shards[shard_id] = ShardNode(shard_id, shard_path)
            except (ValueError, IndexError):
                continue
    
    return shards
