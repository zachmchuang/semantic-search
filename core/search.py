"""High-level search helpers: find candidates similar to a job or profile text."""

from typing import Any

from elasticsearch import Elasticsearch

from core.config import EmbeddingConfig
from core.elasticsearch import knn_search, knn_search_by_job_id
from core.embeddings import embed_texts
from core.indices import CANDIDATE_INDEX, JOB_INDEX

DEFAULT_CANDIDATE_SOURCE = [
    'id',
    'position',
    'primary_keyword',
    'experience_years',
    'english_level',
]


def _format_hits(results: dict) -> list[dict[str, Any]]:
    """Flatten Elasticsearch hit objects into simple dicts with score and fields."""
    formatted = []
    for hit in results['hits']['hits']:
        formatted.append(
            {
                'score': hit['_score'],
                'id': hit['_source'].get('id'),
                **hit['_source'],
            }
        )
    return formatted


def search_candidates_for_job(
    es: Elasticsearch,
    job_embedding: list[float],
    *,
    candidate_index: str = CANDIDATE_INDEX,
    k: int = 10,
    num_candidates: int = 100,
    filters: list[dict] | None = None,
    source: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return top-k candidates whose embeddings are closest to job_embedding."""
    results = knn_search(
        es,
        candidate_index,
        job_embedding,
        k=k,
        num_candidates=num_candidates,
        filters=filters,
        source=source or DEFAULT_CANDIDATE_SOURCE,
    )
    return _format_hits(results)


def search_candidates_by_job_id(
    es: Elasticsearch,
    job_id: str,
    *,
    job_index: str = JOB_INDEX,
    candidate_index: str = CANDIDATE_INDEX,
    k: int = 10,
    num_candidates: int = 100,
    filters: list[dict] | None = None,
    source: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return top-k candidates for a job already indexed in Elasticsearch."""
    results = knn_search_by_job_id(
        es,
        candidate_index,
        job_index,
        job_id,
        k=k,
        num_candidates=num_candidates,
        filters=filters,
        source=source or DEFAULT_CANDIDATE_SOURCE,
    )
    return _format_hits(results)


def search_candidates_by_profile_text(
    profile_text: str,
    es: Elasticsearch,
    *,
    candidate_index: str = CANDIDATE_INDEX,
    k: int = 10,
    num_candidates: int = 100,
    filters: list[dict] | None = None,
    source: list[str] | None = None,
    config: EmbeddingConfig | None = None,
) -> list[dict[str, Any]]:
    """Embed ad-hoc profile text, then search candidates by that vector."""
    query_vec = embed_texts([profile_text], config)[0]
    return search_candidates_for_job(
        es,
        query_vec,
        candidate_index=candidate_index,
        k=k,
        num_candidates=num_candidates,
        filters=filters,
        source=source,
    )
