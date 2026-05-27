"""Coordinator Node: Routes queries to shards and merges results."""

from pathlib import Path
from typing import Optional

import numpy as np
from shard_nodes.shard_node import load_all_shards, ShardNode


class Coordinator:
    """Coordinator node for distributed vector database."""
    
    def __init__(self, shards_dir: str):
        """
        Initialize coordinator.
        
        Args:
            shards_dir: Directory containing all shards
        """
        self.shards_dir = shards_dir
        self.shards: dict[int, ShardNode] = {}
        self.num_shards = 0
        
        self.load_shards()
    
    def load_shards(self) -> None:
        """Load all shard nodes."""
        print("Loading shard nodes...")
        self.shards = load_all_shards(self.shards_dir)
        self.num_shards = len(self.shards)
        if self.num_shards == 0:
            raise FileNotFoundError(f"No shard directories found in {self.shards_dir}")

        print(f"Loaded {self.num_shards} shards")
        
        for shard_id, shard in self.shards.items():
            info = shard.get_info()
            print(f"  Shard {shard_id}: {info['vector_count']} vectors")
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        num_results_per_shard: int = 20,
    ) -> list[tuple[str, str, float]]:
        """
        Perform distributed search across all shards.
        
        Args:
            query_vector: Query vector
            k: Number of final results to return
            num_results_per_shard: Number of results to retrieve from each shard
            
        Returns:
            List of (shard_id, document_id, similarity_score) tuples, sorted by score
        """
        all_results = []
        
        for shard_id, shard in self.shards.items():
            shard_results = shard.search(query_vector, num_results_per_shard)
            
            for doc_id, score in shard_results:
                all_results.append((str(shard_id), doc_id, score))
        
        all_results.sort(key=lambda x: x[2], reverse=True)
        return all_results[:k]
    
    def get_document(self, shard_id: int, doc_id: str) -> tuple[Optional[str], Optional[str]]:
        """
        Retrieve document content from specific shard.
        
        Args:
            shard_id: ID of shard
            doc_id: ID of document
            
        Returns:
            Tuple of (doc_id, document_content)
        """
        if shard_id not in self.shards:
            return None, None
        
        shard = self.shards[shard_id]
        
        if doc_id not in shard.ids:
            return None, None
        
        idx = shard.ids.index(doc_id)
        return doc_id, shard.documents[idx]
    
    def get_status(self) -> dict:
        """Get coordinator status."""
        total_vectors = sum(len(shard.vectors) for shard in self.shards.values())
        
        return {
            "num_shards": self.num_shards,
            "total_vectors": total_vectors,
            "shards": [shard.get_info() for shard in self.shards.values()]
        }


def main():
    """Test coordinator functionality."""
    print("=" * 60)
    print("Distributed Vector Database - Coordinator")
    print("=" * 60)
    
    shards_dir = Path(__file__).resolve().parents[1] / "data" / "shards"
    coordinator = Coordinator(shards_dir)
    
    # Print status
    print("\nCoordinator Status:")
    status = coordinator.get_status()
    print(f"  Total shards: {status['num_shards']}")
    print(f"  Total vectors: {status['total_vectors']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
