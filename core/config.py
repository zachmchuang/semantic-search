import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / '.env')


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    api_url: str = 'https://qwen-embed-runpod.boliu.info/v1/embeddings'
    model: str = 'Qwen/Qwen3-Embedding-0.6B'
    dim: int = 1024
    batch_size: int = 32
    workers: int = 4

    @classmethod
    def from_env(cls, **overrides) -> 'EmbeddingConfig':
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
class ElasticsearchConfig:
    url: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> 'ElasticsearchConfig':
        password = os.environ.get('ES_PASSWORD')
        if not password:
            raise ValueError('ES_PASSWORD not set; add it to .env')
        return cls(
            url=os.environ.get('ES_URL', 'https://localhost:9200'),
            user=os.environ.get('ES_USER', 'elastic'),
            password=password,
        )
