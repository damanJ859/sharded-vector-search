"""Example queries and demonstrations of the Distributed Vector Database."""

import os

import numpy as np
from coordinator.coordinator import Coordinator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHARDS_DIR = os.environ.get("DVD_SHARDS_DIR", os.path.join(BASE_DIR, "data", "shards"))
VECTOR_DIMENSION = int(os.environ.get("DVD_VECTOR_DIMENSION", "1536"))


def example_text_search():
    """Example: Search for documents similar to a query text."""
    print("\n" + "=" * 60)
    print("Example 1: Vector Search")
    print("=" * 60)
    
    coordinator = Coordinator(SHARDS_DIR)
    
    query_embedding = np.random.randn(VECTOR_DIMENSION).astype(np.float32)
    
    print(f"Generated embedding dimension: {len(query_embedding)}")
    
    # Search
    print("\nSearching across all shards...")
    results = coordinator.search(query_embedding, k=5, num_results_per_shard=10)
    
    print(f"\nTop 5 Results:")
    print("-" * 60)
    for i, (shard_id, doc_id, score) in enumerate(results, 1):
        shard_id_int = int(shard_id)
        _, document = coordinator.get_document(shard_id_int, doc_id)
        
        print(f"\n{i}. Similarity Score: {score:.4f} (Shard {shard_id})")
        print(f"   Document ID: {doc_id}")
        print(f"   Preview: {document[:150]}...")


def example_random_search():
    """Example: Search using a random vector (shows distribution)."""
    print("\n" + "=" * 60)
    print("Example 2: Random Vector Search")
    print("=" * 60)
    
    coordinator = Coordinator(SHARDS_DIR)
    
    # Create random query vector
    query_vector = np.random.randn(VECTOR_DIMENSION).astype(np.float32)
    
    print(f"\nQuery: Random vector (dimension: {len(query_vector)})")
    print(f"Vector norm: {np.linalg.norm(query_vector):.4f}")
    
    # Search
    print("\nSearching across all shards...")
    results = coordinator.search(query_vector, k=5, num_results_per_shard=15)
    
    print(f"\nTop 5 Results:")
    print("-" * 60)
    
    shard_distribution = {}
    for i, (shard_id, doc_id, score) in enumerate(results, 1):
        shard_id_int = int(shard_id)
        shard_distribution[shard_id] = shard_distribution.get(shard_id, 0) + 1
        _, document = coordinator.get_document(shard_id_int, doc_id)
        
        print(f"\n{i}. Similarity Score: {score:.4f} (Shard {shard_id})")
        print(f"   Document ID: {doc_id}")
        print(f"   Preview: {document[:150]}...")
    
    print("\n\nShard Distribution in Results:")
    for shard_id in sorted(shard_distribution.keys()):
        print(f"  Shard {shard_id}: {shard_distribution[shard_id]} results")


def example_system_overview():
    """Example: Get system overview and statistics."""
    print("\n" + "=" * 60)
    print("Example 3: System Overview and Statistics")
    print("=" * 60)
    
    coordinator = Coordinator(SHARDS_DIR)
    
    status = coordinator.get_status()
    
    print(f"\nSystem Status:")
    print(f"  Number of Shards: {status['num_shards']}")
    print(f"  Total Vectors: {status['total_vectors']}")
    
    print(f"\nShard Details:")
    for shard_info in status['shards']:
        print(f"  Shard {shard_info['shard_id']}:")
        print(f"    Vectors: {shard_info['vector_count']}")
        print(f"    Dimension: {shard_info['dimension']}")
        print(f"    Percentage: {shard_info['vector_count']/status['total_vectors']*100:.1f}%")


def example_specific_document_retrieval():
    """Example: Retrieve a specific document by ID."""
    print("\n" + "=" * 60)
    print("Example 4: Retrieve Specific Document")
    print("=" * 60)
    
    coordinator = Coordinator(SHARDS_DIR)
    
    # Get first document from shard 0
    shard_id = sorted(coordinator.shards.keys())[0]
    shard = coordinator.shards[shard_id]
    first_doc_id = shard.ids[0]
    
    print(f"\nRetrieving document:")
    print(f"  Shard ID: {shard_id}")
    print(f"  Document ID: {first_doc_id}")
    
    doc_id, document = coordinator.get_document(shard_id, first_doc_id)
    
    print(f"\nDocument Content (first 300 characters):")
    print("-" * 60)
    print(document[:300])
    print("...")


def example_performance_analysis():
    """Example: Analyze search latency across shards."""
    print("\n" + "=" * 60)
    print("Example 5: Performance Analysis")
    print("=" * 60)
    
    import time
    
    coordinator = Coordinator(SHARDS_DIR)
    
    # Create random queries
    num_queries = 10
    query_vectors = [np.random.randn(VECTOR_DIMENSION).astype(np.float32) for _ in range(num_queries)]
    
    print(f"\nRunning {num_queries} random queries...")
    
    times = []
    for i, query_vector in enumerate(query_vectors):
        start = time.time()
        results = coordinator.search(query_vector, k=10)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Query {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\nPerformance Statistics:")
    print(f"  Average Latency: {np.mean(times)*1000:.2f}ms")
    print(f"  Min Latency: {np.min(times)*1000:.2f}ms")
    print(f"  Max Latency: {np.max(times)*1000:.2f}ms")
    print(f"  Std Dev: {np.std(times)*1000:.2f}ms")


def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("Distributed Vector Database - Query Examples")
    print("=" * 60)
    
    try:
        example_system_overview()
        example_text_search()
        example_random_search()
        example_specific_document_retrieval()
        example_performance_analysis()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
