#!/usr/bin/env python3
"""Benchmark embedding API throughput across batch sizes."""

import argparse
import os
import sys
import time
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
BATCH_SIZES = [8, 16, 32, 64, 128]

headers = {
    "Authorization": f"Bearer {EMBED_API_KEY}",
    "Content-Type": "application/json",
}


class EmbedError(Exception):
    """Raised when the embedding API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call the embedding API for one batch of texts."""
    response = requests.post(
        EMBED_API_URL,
        headers=headers,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=300,
    )
    if not response.ok:
        raise EmbedError(response.status_code, response.text[:200])
    data = sorted(response.json()["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def run_benchmark(texts: list[str], batch_size: int, outer: tqdm | None = None) -> dict:
    """Embed all texts at the given batch size and return timing stats."""
    batch_starts = list(range(0, len(texts), batch_size))
    start = time.perf_counter()
    docs_done = 0
    error: str | None = None

    inner = tqdm(
        batch_starts,
        desc=f"batch={batch_size:>3}",
        unit="batch",
        file=sys.stderr,
        leave=False,
        dynamic_ncols=True,
    )
    for offset in inner:
        chunk = texts[offset : offset + batch_size]
        try:
            embed_texts(chunk)
        except EmbedError as exc:
            error = f"HTTP {exc.status_code}"
            inner.set_postfix(error=error)
            if outer is not None:
                outer.set_postfix(error=error)
            break
        docs_done += len(chunk)
        elapsed = time.perf_counter() - start
        docs_per_sec = docs_done / elapsed if elapsed > 0 else 0.0
        inner.set_postfix(docs=f"{docs_done}/{len(texts)}", speed=f"{docs_per_sec:.1f} docs/s")
        if outer is not None:
            outer.set_postfix(current=f"batch={batch_size}", speed=f"{docs_per_sec:.1f} docs/s")

    elapsed = time.perf_counter() - start
    docs_per_sec = docs_done / elapsed if elapsed > 0 and docs_done else 0.0
    return {
        "batch_size": batch_size,
        "batches": len(batch_starts),
        "elapsed_s": elapsed,
        "docs_per_sec": docs_per_sec,
        "docs_done": docs_done,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for doc count and batch sizes to test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=NUM_DOCS)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=BATCH_SIZES,
        help="batch sizes to benchmark (default: 8 16 32 64 128)",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def format_result(result: dict, total_docs: int) -> str:
    """Format one benchmark row for terminal output."""
    if result["error"]:
        return (
            f"batch={result['batch_size']:>3}  FAILED ({result['error']}) "
            f"after {result['docs_done']}/{total_docs} docs"
        )
    return (
        f"batch={result['batch_size']:>3}  "
        f"{result['elapsed_s']:>7.2f}s  "
        f"{result['docs_per_sec']:>7.1f} docs/s  "
        f"({result['batches']} API batches)"
    )


def main() -> None:
    """Run warmup, then benchmark each batch size and print a summary."""
    global NUM_DOCS
    args = parse_args()
    NUM_DOCS = args.num_docs
    batch_sizes = args.batch_sizes

    df = pq.read_table(DATASET_PATH).to_pandas()
    texts = df["profile_text"].head(NUM_DOCS).tolist()
    if len(texts) < NUM_DOCS:
        raise SystemExit(f"Dataset has only {len(texts)} rows, need {NUM_DOCS}")

    tqdm.write(f"Benchmarking {NUM_DOCS} docs via {EMBED_API_URL}")
    tqdm.write(f"Model: {EMBED_MODEL}")
    tqdm.write(f"Batch sizes: {batch_sizes}")
    tqdm.write("")

    if not args.skip_warmup:
        with tqdm(total=8, desc="warmup", unit="doc", file=sys.stderr, dynamic_ncols=True) as warmup:
            embed_texts(texts[:8])
            warmup.update(8)
        tqdm.write("Warmup complete.\n")

    results = []
    with tqdm(
        batch_sizes,
        desc="batch sizes",
        unit="test",
        file=sys.stderr,
        dynamic_ncols=True,
    ) as outer:
        for batch_size in outer:
            outer.set_description(f"testing batch={batch_size}")
            result = run_benchmark(texts, batch_size, outer=outer)
            results.append(result)
            tqdm.write(format_result(result, NUM_DOCS))
            successful = [r for r in results if not r["error"]]
            if successful:
                outer.set_postfix(
                    best_so_far=max(successful, key=lambda r: r["docs_per_sec"])["batch_size"]
                )

    successful = [r for r in results if not r["error"] and r["docs_done"] == len(texts)]
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write("Results (successful runs, sorted by elapsed time):")
    if not successful:
        tqdm.write("  No batch size completed successfully.")
    else:
        fastest = min(successful, key=lambda r: r["elapsed_s"])
        for r in sorted(successful, key=lambda x: x["elapsed_s"]):
            marker = " <-- FASTEST" if r["batch_size"] == fastest["batch_size"] else ""
            tqdm.write(
                f"  batch={r['batch_size']:>3}  "
                f"{r['elapsed_s']:>7.2f}s  "
                f"{r['docs_per_sec']:>7.1f} docs/s{marker}"
            )
        tqdm.write("")
        tqdm.write(
            f"Fastest: batch size {fastest['batch_size']} "
            f"({fastest['elapsed_s']:.2f}s, {fastest['docs_per_sec']:.1f} docs/s)"
        )

    failed = [r for r in results if r["error"]]
    if failed:
        tqdm.write("")
        tqdm.write("Failed batch sizes:")
        for r in failed:
            tqdm.write(f"  batch={r['batch_size']:>3}  {r['error']}")


if __name__ == "__main__":
    main()
