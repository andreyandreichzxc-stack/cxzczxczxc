"""Hybrid search: Reciprocal Rank Fusion (RRF) for keyword + vector results.

Combines BM25/FTS5 keyword results with cosine similarity vector results
using the RRF formula from Cormack et al., SIGIR 2009.

score(d) = sum_{r in rankings} 1 / (k + rank(d, r))

where:
- rank(d, r) is the position (1-indexed) of document d in ranking r
- k = 60 is the smoothing constant
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# RRF smoothing constant — standard value from the literature (Cormack et al. 2009)
RRF_K: int = 60


def reciprocal_rank_fusion(
    vector_results: list[tuple[int, float]] | None = None,
    keyword_results: list[tuple[int, float]] | None = None,
    *,
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Combine two ranked result lists via Reciprocal Rank Fusion.

    Each input is a list of (id, score) tuples, sorted by relevance
    (best first). The score is used as a weight multiplier in RRF so that
    higher-relevance results contribute more, even at the same position.

    Args:
        vector_results: Ranked list of (memory_id, cosine_score) from Qdrant.
        keyword_results: Ranked list of (memory_id, bm25_score) from FTS5.
        k: RRF smoothing constant (default 60).

    Returns:
        List of (memory_id, fused_rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}

    for ranking in (vector_results, keyword_results):
        if not ranking:
            continue
        for rank_i, (mem_id, score) in enumerate(ranking, start=1):
            rrf_contrib = 1.0 / (k + rank_i)
            scores[mem_id] = scores.get(mem_id, 0.0) + rrf_contrib

    # Sort by fused score descending
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


__all__ = [
    "reciprocal_rank_fusion",
    "RRF_K",
]
