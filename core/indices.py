"""Elasticsearch index names and field mappings for candidates and jobs."""

from core.config import VectorIndexConfig

CANDIDATE_INDEX = 'candidate_profiles'
JOB_INDEX = 'job_profiles'


def candidate_mappings(vector_config: VectorIndexConfig | None = None) -> dict:
    """Field schema for the candidate index, including a searchable embedding vector."""
    cfg = vector_config or VectorIndexConfig.from_env()
    return {
        'properties': {
            'id': {'type': 'keyword'},
            'profile_text': {'type': 'text'},
            'position': {'type': 'text'},
            'english_level': {'type': 'keyword'},
            'experience_years': {'type': 'float'},
            'primary_keyword': {'type': 'keyword'},
            'cv_lang': {'type': 'keyword'},
            'embedding': cfg.to_vector_field(),
        }
    }


def job_mappings(vector_config: VectorIndexConfig | None = None) -> dict:
    """Field schema for the job index, including a searchable embedding vector."""
    cfg = vector_config or VectorIndexConfig.from_env()
    return {
        'properties': {
            'id': {'type': 'keyword'},
            'profile_text': {'type': 'text'},
            'position': {'type': 'text'},
            'description': {'type': 'text'},
            'company_name': {'type': 'keyword'},
            'exp_years': {'type': 'keyword'},
            'primary_keyword': {'type': 'keyword'},
            'english_level': {'type': 'keyword'},
            'published': {'type': 'keyword'},
            'description_lang': {'type': 'keyword'},
            'embedding': cfg.to_vector_field(),
        }
    }
