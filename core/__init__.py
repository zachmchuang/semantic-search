"""Public API for semantic search: embeddings, profiles, and candidate search."""

from core.config import EmbeddingConfig, ElasticsearchConfig, VectorIndexConfig
from core.embeddings import embed_all, embed_texts
from core.profiles import row_to_candidate_profile, row_to_job_profile
from core.search import search_candidates_by_job_id, search_candidates_for_job

__all__ = [
    'EmbeddingConfig',
    'ElasticsearchConfig',
    'VectorIndexConfig',
    'embed_all',
    'embed_texts',
    'row_to_candidate_profile',
    'row_to_job_profile',
    'search_candidates_by_job_id',
    'search_candidates_for_job',
]
