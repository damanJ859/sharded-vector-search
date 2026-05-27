"""Vector extraction and sharding script for Distributed Vector Database."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from utils import assign_to_shard


def extract_vectors_from_chroma(
    chroma_path: str, 
    collection_name: str = "langchain"
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Extract all vectors, IDs, and documents from Chroma vector store.
    
    Args:
        chroma_path: Path to Chroma persistent storage
        collection_name: Name of collection to extract
        
    Returns:
        Tuple of (vectors array, ids list, documents list)
    """
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    
    # Get all data
    all_data = collection.get(include=['embeddings', 'documents'])
    
    vectors = np.array(all_data['embeddings'], dtype=np.float32)
    ids = all_data['ids']
    documents = all_data['documents']

    if vectors.size == 0:
        raise ValueError(f"No embeddings found in Chroma collection '{collection_name}'")
    
    print(f"Extracted {len(vectors)} vectors with dimension {vectors.shape[1]}")
    return vectors, ids, documents


def shard_vectors(
    vectors: np.ndarray, 
    ids: list[str], 
    documents: list[str],
    num_shards: int = 3,
    lsh_bits: int = 10,
    seed: int = 42,
) -> dict[int, dict[str, Any]]:
    """
    Distribute vectors across shards using LSH hashing.
    
    Args:
        vectors: numpy array of vectors
        ids: list of vector IDs
        documents: list of documents
        num_shards: number of shards to create
        
    Returns:
        Dictionary mapping shard_id -> {vectors, ids, documents}
    """
    shards: dict[int, dict[str, Any]] = {
        i: {"vectors": [], "ids": [], "documents": []} for i in range(num_shards)
    }
    
    for i, (vector, vec_id, doc) in enumerate(zip(vectors, ids, documents)):
        shard_id = assign_to_shard(vector, num_shards, num_bits=lsh_bits, seed=seed)
        shards[shard_id]["vectors"].append(vector)
        shards[shard_id]["ids"].append(vec_id)
        shards[shard_id]["documents"].append(doc)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(vectors)} vectors")
    
    # Convert lists to numpy arrays
    for shard_id in shards:
        shards[shard_id]["vectors"] = np.array(shards[shard_id]["vectors"])
    
    return shards


def save_shards(shards: dict[int, dict[str, Any]], output_dir: str) -> None:
    """Save shard data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    shard_info = {}
    for shard_id, shard_data in shards.items():
        shard_path = os.path.join(output_dir, f"shard_{shard_id}")
        os.makedirs(shard_path, exist_ok=True)
        
        # Save vectors
        vectors_path = os.path.join(shard_path, "vectors.npy")
        np.save(vectors_path, shard_data["vectors"])
        
        # Save metadata (IDs and documents)
        metadata = {
            "ids": shard_data["ids"],
            "documents": shard_data["documents"]
        }
        metadata_path = os.path.join(shard_path, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        shard_info[shard_id] = {
            "count": len(shard_data["ids"]),
            "dimension": shard_data["vectors"].shape[1]
        }
        
        print(f"Saved shard {shard_id}: {shard_info[shard_id]['count']} vectors")
    
    # Save shard info
    info_path = os.path.join(output_dir, "shard_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(shard_info, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Chroma vectors and build local shards.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--chroma-path",
        default=os.getenv("DVD_CHROMA_PATH", str(repo_root / "Vec_Store")),
        help="Path to the Chroma persistent store.",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("DVD_CHROMA_COLLECTION", "langchain"),
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("DVD_SHARDS_DIR", str(Path(__file__).resolve().parent / "data" / "shards")),
        help="Directory where shard files will be written.",
    )
    parser.add_argument("--num-shards", type=int, default=3, help="Number of shards to create.")
    parser.add_argument("--lsh-bits", type=int, default=10, help="Number of projection bits for LSH.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic LSH projections.")
    return parser.parse_args()


def main() -> None:
    """Main extraction and sharding pipeline."""
    args = parse_args()

    print("=" * 60)
    print("Vector Extraction and Sharding Pipeline")
    print("=" * 60)
    
    print("\n[1] Extracting vectors from Chroma...")
    vectors, ids, documents = extract_vectors_from_chroma(args.chroma_path, args.collection)
    
    print(f"\n[2] Sharding vectors across {args.num_shards} nodes using LSH...")
    shards = shard_vectors(
        vectors,
        ids,
        documents,
        num_shards=args.num_shards,
        lsh_bits=args.lsh_bits,
        seed=args.seed,
    )
    
    # Display shard distribution
    print("\nShard Distribution:")
    for shard_id in range(args.num_shards):
        count = len(shards[shard_id]["ids"])
        print(f"  Shard {shard_id}: {count} vectors ({count/len(vectors)*100:.1f}%)")
    
    print("\n[3] Saving shards to disk...")
    save_shards(shards, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Extraction and sharding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
