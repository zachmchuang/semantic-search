"""Configuration loaded from environment variables (.env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of the core/ package).
load_dotenv(Path(__file__).resolve().parent.parent / '.env')


@dataclass(frozen=True)
class EmbeddingConfig:
    """Settings for the embedding API (model, dimensions, batching)."""

    api_key: str
    api_url: str = 'https://qwen-embed-runpod.boliu.info/v1/embeddings'
    model: str = 'Qwen/Qwen3-Embedding-0.6B'
    dim: int = 1024
    batch_size: int = 32
    workers: int = 4

    @classmethod
    def from_env(cls, **overrides) -> 'EmbeddingConfig':
        """Build config from env vars; pass keyword args to override defaults."""
        api_key = os.environ.get('EMBED_API_KEY')
        if not api_key:
            raise ValueError('EMBED_API_KEY not set; add it to .env')
        defaults = {
            'api_key': api_key,
            'api_url': os.environ.get('EMBED_API_URL', cls.api_url),
            'model': os.environ.get('EMBED_MODEL', cls.model),
            'dim': int(os.environ.get('EMBEDDING_DIM', cls.dim)),
            'batch_size': int(os.environ.get('EMBED_BATCH_SIZE', cls.batch_size)),
            'workers': int(os.environ.get('EMBED_WORKERS', cls.workers)),
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass(frozen=True)
class VectorIndexConfig:
    """Elasticsearch dense_vector and index settings for kNN search.

    Edit the defaults below, or override via environment variables. Mapping
    changes require deleting and re-creating the index (re-run the embedding
    notebooks with existing .npy embeddings).
    """

    dim: int = 1024
    index: bool = True
    similarity: str = 'cosine'
    index_options_type: str = 'int8_hnsw'
    index_options_m: int = 16
    index_options_ef_construction: int = 100
    exclude_source_vectors: bool = False
    number_of_shards: int | None = None
    number_of_replicas: int | None = None

    @classmethod
    def from_env(cls, **overrides) -> 'VectorIndexConfig':
        """Build config from env vars; pass keyword args to override defaults."""
        defaults = {
            'dim': int(os.environ.get('EMBEDDING_DIM', cls.dim)),
            'index': _env_bool('ES_VECTOR_INDEX', cls.index),
            'similarity': os.environ.get('ES_VECTOR_SIMILARITY', cls.similarity),
            'index_options_type': os.environ.get(
                'ES_VECTOR_INDEX_OPTIONS_TYPE',
                cls.index_options_type,
            ),
            'index_options_m': int(
                os.environ.get('ES_VECTOR_INDEX_OPTIONS_M', cls.index_options_m)
            ),
            'index_options_ef_construction': int(
                os.environ.get(
                    'ES_VECTOR_INDEX_OPTIONS_EF_CONSTRUCTION',
                    cls.index_options_ef_construction,
                )
            ),
            'exclude_source_vectors': _env_bool(
                'ES_EXCLUDE_SOURCE_VECTORS',
                cls.exclude_source_vectors,
            ),
            'number_of_shards': _optional_int('ES_NUMBER_OF_SHARDS'),
            'number_of_replicas': _optional_int('ES_NUMBER_OF_REPLICAS'),
        }
        defaults.update(overrides)
        return cls(**defaults)

    def to_vector_field(self) -> dict:
        """Elasticsearch mapping for the ``embedding`` dense_vector field."""
        field: dict = {
            'type': 'dense_vector',
            'dims': self.dim,
            'index': self.index,
        }
        if not self.index:
            return field

        field['similarity'] = self.similarity
        index_options = self._index_options()
        if index_options:
            field['index_options'] = index_options
        return field

    def to_index_settings(self) -> dict:
        """Elasticsearch index settings applied at create time."""
        settings: dict = {
            'index.mapping.exclude_source_vectors': self.exclude_source_vectors,
        }
        if self.number_of_shards is not None:
            settings['number_of_shards'] = self.number_of_shards
        if self.number_of_replicas is not None:
            settings['number_of_replicas'] = self.number_of_replicas
        return settings

    def _index_options(self) -> dict | None:
        options: dict = {}
        if self.index_options_type is not None:
            options['type'] = self.index_options_type
        if self.index_options_m is not None:
            options['m'] = self.index_options_m
        if self.index_options_ef_construction is not None:
            options['ef_construction'] = self.index_options_ef_construction
        return options or None


def _optional_str(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None or value.strip() == '':
        return None
    return value


def _optional_int(key: str) -> int | None:
    value = os.environ.get(key)
    if value is None or value.strip() == '':
        return None
    return int(value)


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None or value.strip() == '':
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class ElasticsearchConfig:
    """Connection settings for the Elasticsearch cluster."""

    url: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> 'ElasticsearchConfig':
        """Build config from ES_URL, ES_USER, and ES_PASSWORD env vars."""
        password = os.environ.get('ES_PASSWORD')
        if not password:
            raise ValueError('ES_PASSWORD not set; add it to .env')
        return cls(
            url=os.environ.get('ES_URL', 'https://localhost:9200'),
            user=os.environ.get('ES_USER', 'elastic'),
            password=password,
        )
