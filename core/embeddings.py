import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from tqdm.auto import tqdm

from core.config import EmbeddingConfig

_thread_local = threading.local()


def _get_session(config: EmbeddingConfig) -> requests.Session:
    session = getattr(_thread_local, 'session', None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                'Authorization': f'Bearer {config.api_key}',
                'Content-Type': 'application/json',
            }
        )
        _thread_local.session = session
    return session


def embed_texts(texts: list[str], config: EmbeddingConfig | None = None) -> list[list[float]]:
    cfg = config or EmbeddingConfig.from_env()
    response = _get_session(cfg).post(
        cfg.api_url,
        json={'model': cfg.model, 'input': texts},
        timeout=300,
    )
    response.raise_for_status()
    data = sorted(response.json()['data'], key=lambda item: item['index'])
    return [item['embedding'] for item in data]


def load_completed_starts(progress_path: Path, total: int, batch_size: int) -> set[int]:
    if not progress_path.exists():
        return set()
    progress = json.loads(progress_path.read_text())
    if progress.get('total') != total:
        return set()
    if 'completed_starts' in progress:
        return set(progress['completed_starts'])
    next_offset = int(progress.get('next_offset', 0))
    return set(range(0, next_offset, batch_size))


def contiguous_offset(completed_starts: set[int], batch_size: int, total: int) -> int:
    offset = 0
    while offset < total and offset in completed_starts:
        offset += batch_size
    return offset


def save_progress(
    progress_path: Path,
    total: int,
    completed_starts: set[int],
    batch_size: int,
) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                'total': total,
                'next_offset': contiguous_offset(completed_starts, batch_size, total),
                'completed_starts': sorted(completed_starts),
            },
            indent=2,
        )
    )


def _embed_batch(
    job: tuple[int, list[str]],
    config: EmbeddingConfig,
) -> tuple[int, list[list[float]]]:
    start, chunk = job
    return start, embed_texts(chunk, config)


def embed_all(
    texts: list[str],
    embeddings_path: Path,
    progress_path: Path,
    config: EmbeddingConfig | None = None,
) -> np.ndarray:
    cfg = config or EmbeddingConfig.from_env()
    total = len(texts)
    completed_starts = load_completed_starts(progress_path, total, cfg.batch_size)
    resume_offset = contiguous_offset(completed_starts, cfg.batch_size, total)

    if not embeddings_path.exists():
        emb = np.memmap(
            embeddings_path,
            dtype=np.float32,
            mode='w+',
            shape=(total, cfg.dim),
        )
    else:
        emb = np.memmap(
            embeddings_path,
            dtype=np.float32,
            mode='r+',
            shape=(total, cfg.dim),
        )

    pending_jobs = [
        (start, texts[start : start + cfg.batch_size])
        for start in range(0, total, cfg.batch_size)
        if start not in completed_starts
    ]

    print(f'Embedding {total:,} profiles via {cfg.api_url}')
    print(
        f'Model: {cfg.model}  |  batch size: {cfg.batch_size}  |  workers: {cfg.workers}'
    )
    print(f'Resuming from offset: {resume_offset:,}  |  pending batches: {len(pending_jobs):,}')

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [
            pool.submit(_embed_batch, job, cfg) for job in pending_jobs
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc='API batches',
            unit='batch',
        ):
            start, vectors = future.result()
            end = start + len(vectors)
            emb[start:end] = vectors
            emb.flush()
            completed_starts.add(start)
            save_progress(progress_path, total, completed_starts, cfg.batch_size)

    save_progress(
        progress_path,
        total,
        set(range(0, total, cfg.batch_size)),
        cfg.batch_size,
    )
    result = np.asarray(emb)
    print(f'Done. Embeddings shape: {result.shape}')
    return result
