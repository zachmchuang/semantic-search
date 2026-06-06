from core.config import EmbeddingConfig

CANDIDATE_INDEX = 'candidate_profiles'
JOB_INDEX = 'job_profiles'


def candidate_mappings(config: EmbeddingConfig | None = None) -> dict:
    dim = (config or EmbeddingConfig.from_env()).dim
    return {
        'properties': {
            'id': {'type': 'keyword'},
            'profile_text': {'type': 'text'},
            'position': {'type': 'text'},
            'english_level': {'type': 'keyword'},
            'experience_years': {'type': 'float'},
            'primary_keyword': {'type': 'keyword'},
            'cv_lang': {'type': 'keyword'},
            'embedding': _vector_field(dim),
        }
    }


def job_mappings(config: EmbeddingConfig | None = None) -> dict:
    dim = (config or EmbeddingConfig.from_env()).dim
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
            'embedding': _vector_field(dim),
        }
    }


def _vector_field(dim: int) -> dict:
    return {
        'type': 'dense_vector',
        'dims': dim,
        'index': True,
        'similarity': 'cosine',
    }
