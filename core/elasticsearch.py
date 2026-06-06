from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, NotFoundError, helpers

from core.config import ElasticsearchConfig
from core.indices import CANDIDATE_INDEX, JOB_INDEX
from core.profiles import field_value


def get_es_client(config: ElasticsearchConfig | None = None) -> Elasticsearch:
    cfg = config or ElasticsearchConfig.from_env()
    return Elasticsearch(
        cfg.url,
        basic_auth=(cfg.user, cfg.password),
        verify_certs=False,
    )


def create_index(
    es: Elasticsearch,
    index: str,
    mappings: dict,
    *,
    recreate: bool = True,
) -> None:
    if recreate:
        try:
            es.indices.delete(index=index)
            print(f'Deleted existing index: {index}')
        except NotFoundError:
            pass

    es.indices.create(index=index, mappings=mappings)
    print(f'Created index: {index}')


def bulk_index(
    es: Elasticsearch,
    index: str,
    actions: Iterator[dict],
    *,
    chunk_size: int = 500,
) -> tuple[int, list]:
    success, errors = helpers.bulk(
        es,
        actions,
        chunk_size=chunk_size,
        request_timeout=120,
        raise_on_error=False,
    )
    es.indices.refresh(index=index)
    print(f'Indexed documents: {success:,}')
    if errors:
        print(f'Bulk errors: {len(errors)}')
        print(errors[:3])
    return success, errors


def knn_search(
    es: Elasticsearch,
    index: str,
    query_vector: list[float],
    *,
    k: int = 10,
    num_candidates: int = 100,
    filters: list[dict] | None = None,
    source: list[str] | None = None,
) -> dict:
    knn: dict[str, Any] = {
        'field': 'embedding',
        'query_vector': query_vector,
        'k': k,
        'num_candidates': num_candidates,
    }
    if filters:
        knn['filter'] = filters

    return es.search(
        index=index,
        knn=knn,
        _source=source,
    )


def candidate_actions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    index: str = CANDIDATE_INDEX,
    id_column: str = 'id',
) -> Iterator[dict]:
    for i in range(len(df)):
        row = df.iloc[i]
        yield {
            '_index': index,
            '_id': row[id_column],
            '_source': {
                'id': row[id_column],
                'profile_text': row['profile_text'],
                'position': field_value(row, 'Position'),
                'english_level': field_value(row, 'English Level'),
                'experience_years': field_value(row, 'Experience Years', default=None),
                'primary_keyword': field_value(row, 'Primary Keyword'),
                'cv_lang': field_value(row, 'CV_lang'),
                'embedding': embeddings[i].tolist(),
            },
        }


def job_actions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    index: str = JOB_INDEX,
    id_column: str = 'id',
    description_column: str | None = None,
) -> Iterator[dict]:
    desc_col = description_column or _resolve_column(
        df,
        'Long Description',
        'long description',
        'Description',
        'description',
    )

    for i in range(len(df)):
        row = df.iloc[i]
        yield {
            '_index': index,
            '_id': row[id_column],
            '_source': {
                'id': row[id_column],
                'profile_text': row['profile_text'],
                'position': _field_from_row(row, 'Position', 'position'),
                'description': _field_from_row(row, desc_col) if desc_col else '',
                'company_name': _field_from_row(row, 'Company Name', 'company name'),
                'exp_years': _field_from_row(row, 'Exp Years', 'exp years'),
                'primary_keyword': _field_from_row(row, 'Primary Keyword', 'primary keyword'),
                'english_level': _field_from_row(row, 'English Level', 'english level'),
                'published': _field_from_row(row, 'Published', 'published'),
                'description_lang': _field_from_row(
                    row,
                    'Long Description_lang',
                    'long description_lang',
                    'description_lang',
                ),
                'embedding': embeddings[i].tolist(),
            },
        }


def _resolve_column(df: pd.DataFrame, *names: str) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _field_from_row(row: pd.Series, *names: str, default=''):
    for name in names:
        if name in row.index:
            value = row[name]
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            return value
    return default
