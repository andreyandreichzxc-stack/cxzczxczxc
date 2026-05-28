"""Cosine similarity utilities for the memory system.

Shared by: MMR reranking, semantic linking, contradiction detection, graph scoring.
"""

from __future__ import annotations
import math
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns value in [-1, 1]. Returns 0.0 for zero vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_cosine(query: list[float], vectors: list[list[float]]) -> list[float]:
    """Compute cosine similarity between a query vector and multiple vectors.

    Returns list of similarities in the same order as vectors.
    """
    if not query:
        return [0.0] * len(vectors)
    return [cosine_similarity(query, v) for v in vectors]


def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity matrix for a list of vectors.

    Returns NxN matrix where matrix[i][j] = cosine(vectors[i], vectors[j]).
    Used by MMR reranking to compute inter-document diversity.
    """
    n = len(vectors)
    if n == 0:
        return []

    # Precompute norms
    norms = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v))
        norms.append(norm if norm > 0 else 1.0)

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0  # self-similarity
        for j in range(i + 1, n):
            dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
            sim = dot / (norms[i] * norms[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


def top_k_diverse(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidate_scores: list[float],
    k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Select top-k candidates balancing relevance and diversity (MMR).

    Args:
        query_embedding: the query vector
        candidate_embeddings: vectors for each candidate
        candidate_scores: relevance scores (e.g. RRF scores) for each candidate
        k: number of candidates to select
        lambda_param: 0.0 = pure diversity, 1.0 = pure relevance

    Returns:
        List of selected indices (into candidate_embeddings/candidate_scores).
    """
    n = len(candidate_embeddings)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    # Precompute query similarities
    query_sims = batch_cosine(query_embedding, candidate_embeddings)

    # Precompute pairwise similarity matrix
    sim_matrix = cosine_similarity_matrix(candidate_embeddings)

    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_score = -float("inf")
        best_idx = -1

        for i in remaining:
            relevance = candidate_scores[i] if i < len(candidate_scores) else 0.0

            # Max similarity to already selected documents
            max_sim = 0.0
            for j in selected:
                if sim_matrix[i][j] > max_sim:
                    max_sim = sim_matrix[i][j]

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr > best_score:
                best_score = mmr
                best_idx = i

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return selected
