#!/usr/bin/env python3
"""Compare sequential vs concurrent embedding with batch size 32."""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBED_API_KEY = os.environ.get("EMBED_API_KEY")
if not EMBED_API_KEY:
    raise SystemExit("EMBED_API_KEY not set; add it to .env")

DATASET_PATH = Path(__file__).resolve().parent.parent / "data/candidate_profiles.parquet"
EMBED_API_URL = "https://qwen-embed-runpod.boliu.info/v1/embeddings"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
NUM_DOCS = 2000
BATCH_SIZE = 32
WORKER_COUNTS = [1, 2, 3, 4]

HEADERS = {
    "Authorization": f"Bearer {EMBED_API_KEY}",
    "Content-Type": "application/json",
}

_thread_local = threading.local()


class EmbedError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return session


def embed_texts(texts: list[str], session: requests.Session | None = None) -> list[list[float]]:
    client = session or get_session()
    response = client.post(
        EMBED_API_URL,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    if not response.ok:
        raise EmbedError(response.status_code, response.text[:200])
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def make_batches(texts: list[str], batch_size: int) -> list[tuple[int, list[str]]]:
    return [
        (offset, texts[offset : offset + batch_size])
        for offset in range(0, len(texts), batch_size)
    ]


def embed_batch(job: tuple[int, list[str]]) -> tuple[int, int]:
    offset, chunk = job
    embed_texts(chunk)
    return offset, len(chunk)


def run_sequential(texts: list[str], batch_size: int) -> dict:
    batches = make_batches(texts, batch_size)
    session = get_session()
    start = time.perf_counter()
    docs_done = 0
    error: str | None = None

    with tqdm(
        batches,
        desc="sequential",
        unit="batch",
        file=sys.stderr,
        dynamic_ncols=True,
    ) as bar:
        for offset, chunk in bar:
            try:
                embed_texts(chunk, session=session)
            except EmbedError as exc:
                error = f"HTTP {exc.status_code}"
                bar.set_postfix(error=error)
                break
            docs_done += len(chunk)
            elapsed = time.perf_counter() - start
            bar.set_postfix(
                docs=f"{docs_done}/{len(texts)}",
                speed=f"{docs_done / elapsed:.1f} docs/s",
            )

    elapsed = time.perf_counter() - start
    return {
        "mode": "sequential",
        "workers": 1,
        "elapsed_s": elapsed,
        "docs_per_sec": docs_done / elapsed if elapsed > 0 and docs_done else 0.0,
        "docs_done": docs_done,
        "error": error,
    }


def run_concurrent(texts: list[str], batch_size: int, workers: int) -> dict:
    batches = make_batches(texts, batch_size)
    start = time.perf_counter()
    docs_done = 0
    error: str | None = None

    with tqdm(
        total=len(batches),
        desc=f"workers={workers}",
        unit="batch",
        file=sys.stderr,
        dynamic_ncols=True,
    ) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(embed_batch, job) for job in batches]
            for future in as_completed(futures):
                try:
                    _, count = future.result()
                except EmbedError as exc:
                    error = f"HTTP {exc.status_code}"
                    bar.set_postfix(error=error)
                    for pending in futures:
                        pending.cancel()
                    break
                docs_done += count
                elapsed = time.perf_counter() - start
                bar.update(1)
                bar.set_postfix(
                    docs=f"{docs_done}/{len(texts)}",
                    speed=f"{docs_done / elapsed:.1f} docs/s",
                )

    elapsed = time.perf_counter() - start
    return {
        "mode": "concurrent",
        "workers": workers,
        "elapsed_s": elapsed,
        "docs_per_sec": docs_done / elapsed if elapsed > 0 and docs_done else 0.0,
        "docs_done": docs_done,
        "error": error,
    }


def main() -> None:
    df = pq.read_table(DATASET_PATH).to_pandas()
    texts = df["profile_text"].head(NUM_DOCS).tolist()
    if len(texts) < NUM_DOCS:
        raise SystemExit(f"Dataset has only {len(texts)} rows, need {NUM_DOCS}")

    tqdm.write(f"Benchmarking {NUM_DOCS} docs, batch size {BATCH_SIZE}")
    tqdm.write(f"API: {EMBED_API_URL}")
    tqdm.write(f"Worker counts: {WORKER_COUNTS}")
    tqdm.write("")

    with tqdm(total=8, desc="warmup", unit="doc", file=sys.stderr, dynamic_ncols=True) as warmup:
        embed_texts(texts[:8])
        warmup.update(8)
    tqdm.write("Warmup complete.\n")

    results = []
    configs = [(1, lambda t: run_sequential(t, BATCH_SIZE))] + [
        (workers, lambda t, w=workers: run_concurrent(t, BATCH_SIZE, w))
        for workers in WORKER_COUNTS
        if workers > 1
    ]

    with tqdm(configs, desc="configs", unit="test", file=sys.stderr, dynamic_ncols=True) as outer:
        for workers, runner in outer:
            label = "sequential" if workers == 1 else f"{workers} workers"
            outer.set_description(label)
            result = runner(texts)
            results.append(result)
            if result["error"]:
                tqdm.write(
                    f"{label:>12}  FAILED ({result['error']}) "
                    f"after {result['docs_done']}/{NUM_DOCS} docs"
                )
            else:
                tqdm.write(
                    f"{label:>12}  {result['elapsed_s']:>7.2f}s  "
                    f"{result['docs_per_sec']:>7.1f} docs/s"
                )

    successful = [r for r in results if not r["error"] and r["docs_done"] == len(texts)]
    tqdm.write("")
    tqdm.write("=" * 60)
    if not successful:
        tqdm.write("No configuration completed successfully.")
        return

    fastest = max(successful, key=lambda r: r["docs_per_sec"])
    baseline = next(r for r in successful if r["workers"] == 1)
    speedup = fastest["docs_per_sec"] / baseline["docs_per_sec"]

    tqdm.write("Results (sorted by throughput):")
    for result in sorted(successful, key=lambda r: r["docs_per_sec"], reverse=True):
        label = "sequential" if result["workers"] == 1 else f"{result['workers']} workers"
        marker = " <-- FASTEST" if result is fastest else ""
        tqdm.write(
            f"  {label:>12}  {result['elapsed_s']:>7.2f}s  "
            f"{result['docs_per_sec']:>7.1f} docs/s{marker}"
        )
    tqdm.write("")
    tqdm.write(
        f"Best: {('sequential' if fastest['workers'] == 1 else str(fastest['workers']) + ' workers')} "
        f"({fastest['docs_per_sec']:.1f} docs/s, {speedup:.2f}x vs sequential)"
    )


if __name__ == "__main__":
    main()
