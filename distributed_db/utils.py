"""Utility functions for distributed vector database."""

from functools import lru_cache
import json
from typing import Any

import numpy as np


@lru_cache(maxsize=32)
def _projection_matrix(dimension: int, num_bits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((dimension, num_bits))


def lsh_hash(vector: np.ndarray, num_bits: int = 10, seed: int = 42) -> str:
    """
    Generate LSH hash for a vector using random projections.
    Maps vector to a hash bucket for sharding.
    """
    random_matrix = _projection_matrix(len(vector), num_bits, seed)
    projections = np.dot(vector, random_matrix)
    hash_bits = (projections > 0).astype(int)
    hash_str = ''.join(map(str, hash_bits))
    return hash_str


def assign_to_shard(
    vector: np.ndarray,
    num_shards: int = 3,
    num_bits: int = 10,
    seed: int = 42,
) -> int:
    """Assign a vector to a shard based on LSH hash."""
    if num_shards <= 0:
        raise ValueError("num_shards must be greater than zero")

    hash_str = lsh_hash(vector, num_bits, seed)
    hash_int = int(hash_str, 2)
    shard_id = hash_int % num_shards
    return shard_id


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def save_json(data: dict[str, Any], filepath: str) -> None:
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
