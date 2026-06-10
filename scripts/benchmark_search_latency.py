#!/usr/bin/env python3
"""Benchmark kNN search latency: job embedding lookup -> candidate pool."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import urllib3

# Local ES uses self-signed TLS (verify_certs=False in get_es_client).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ElasticsearchConfig, VectorIndexConfig
from core.elasticsearch import get_es_client
from core.indices import CANDIDATE_INDEX, JOB_INDEX
from core.search import search_candidates_by_job_id

DEFAULT_JOB_DATASET = (
    Path(__file__).resolve().parent.parent / 'data/job_profiles.parquet'
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for warmup thresholds and measurement size."""
    parser = argparse.ArgumentParser(
        description='Benchmark job-to-candidate kNN search latency.',
    )
    parser.add_argument('--k', type=int, default=10, help='top-k results (default: 10)')
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=100,
        help='kNN num_candidates (default: 100)',
    )
    parser.add_argument(
        '--measurement-iterations',
        type=int,
        default=1000,
        help='queries to record after warmup (default: 1000)',
    )
    parser.add_argument(
        '--warmup-window-size',
        type=int,
        default=20,
        help='queries per warmup window for p50 tracking (default: 20)',
    )
    parser.add_argument(
        '--warmup-stable-windows',
        type=int,
        default=5,
        help='consecutive stable windows required to finish warmup (default: 5)',
    )
    parser.add_argument(
        '--warmup-tolerance',
        type=float,
        default=0.05,
        help='max relative running-p50 spread to finish warmup (default: 0.05)',
    )
    parser.add_argument(
        '--min-warmup',
        type=int,
        default=50,
        help='minimum warmup queries before checking stability (default: 50)',
    )
    parser.add_argument(
        '--max-warmup',
        type=int,
        default=5000,
        help='warmup query cap (default: 5000)',
    )
    parser.add_argument(
        '--job-dataset',
        type=Path,
        default=DEFAULT_JOB_DATASET,
        help='parquet file used to sample job IDs',
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile of values in milliseconds."""
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def relative_spread(values: list[float]) -> float:
    """Return (max - min) / median; used to detect warmup stabilization."""
    if not values:
        return 0.0
    low = min(values)
    high = max(values)
    mid = float(np.median(values))
    if mid == 0:
        return 0.0
    return (high - low) / mid


def is_stable(running_p50s: list[float], tolerance: float, stable_windows: int) -> bool:
    """True when cumulative p50 over recent windows varies within tolerance."""
    if len(running_p50s) < stable_windows:
        return False
    recent = running_p50s[-stable_windows:]
    return relative_spread(recent) <= tolerance


def load_job_ids(path: Path) -> list[str]:
    """Load job document IDs from parquet for query rotation."""
    if not path.exists():
        raise SystemExit(f'Job dataset not found: {path}')
    table = pq.read_table(path, columns=['id'])
    return [str(value) for value in table.column('id').to_pylist()]


def run_search(
    es,
    job_id: str,
    *,
    k: int,
    num_candidates: int,
) -> tuple[float, bool]:
    """Run one job-to-candidate search and return latency (ms) and success flag."""
    start = time.perf_counter()
    try:
        search_candidates_by_job_id(
            es,
            job_id,
            k=k,
            num_candidates=num_candidates,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, False
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, True


def print_summary(label: str, latencies: list[float], errors: int) -> None:
    """Print percentile and summary stats for a timed query phase."""
    if not latencies:
        print(f'{label}: no successful queries')
        return

    arr = np.array(latencies)
    total = len(latencies) + errors
    print(f'\n{label} ({len(latencies):,} ok, {errors:,} errors, {total:,} total)')
    print(f'  p50:  {percentile(latencies, 50):7.2f} ms')
    print(f'  p90:  {percentile(latencies, 90):7.2f} ms')
    print(f'  p95:  {percentile(latencies, 95):7.2f} ms')
    print(f'  p99:  {percentile(latencies, 99):7.2f} ms')
    print(f'  mean: {arr.mean():7.2f} ms')
    print(f'  std:  {arr.std():7.2f} ms')
    print(f'  min:  {arr.min():7.2f} ms')
    print(f'  max:  {arr.max():7.2f} ms')
    if arr.mean() > 0:
        print(f'  qps:  {1000.0 / arr.mean():7.2f} req/s (mean latency)')


def warmup(
    es,
    job_ids: list[str],
    *,
    k: int,
    num_candidates: int,
    window_size: int,
    stable_windows: int,
    tolerance: float,
    min_warmup: int,
    max_warmup: int,
) -> tuple[list[float], int, int]:
    """Run queries until window p50 stabilizes; return latencies, errors, and job offset."""
    latencies: list[float] = []
    window_latencies: list[float] = []
    running_p50s: list[float] = []
    errors = 0
    job_idx = 0
    attempts = 0

    print('\nWarmup (search_candidates_by_job_id)')
    print('  queries  window_p50   running_p50   running_spread')
    print('  ' + '-' * 55)

    while attempts < max_warmup:
        attempts += 1
        job_id = job_ids[job_idx % len(job_ids)]
        job_idx += 1

        elapsed_ms, ok = run_search(
            es,
            job_id,
            k=k,
            num_candidates=num_candidates,
        )
        if ok:
            latencies.append(elapsed_ms)
            window_latencies.append(elapsed_ms)
        else:
            errors += 1

        if len(window_latencies) < window_size:
            continue

        window_p50 = percentile(window_latencies, 50)
        running_p50 = percentile(latencies, 50)
        running_p50s.append(running_p50)
        # Per-window p50 is noisy; stability uses cumulative running p50 instead.
        spread = relative_spread(running_p50s[-stable_windows:])
        print(
            f'  {len(latencies):6d}  {window_p50:9.2f} ms  {running_p50:9.2f} ms'
            f'  {spread * 100:5.1f}%',
            flush=True,
        )
        window_latencies.clear()

        if (
            len(latencies) >= min_warmup
            and is_stable(running_p50s, tolerance, stable_windows)
        ):
            print(
                f'\nWarmup stable after {len(latencies):,} queries '
                f'(last {stable_windows} running p50 spread '
                f'{relative_spread(running_p50s[-stable_windows:]) * 100:.1f}% '
                f'<= {tolerance * 100:.1f}%)',
            )
            break
    else:
        print(
            f'\nWarmup stopped at max {max_warmup:,} queries '
            f'without reaching stability threshold',
        )

    return latencies, errors, job_idx


def measure(
    es,
    job_ids: list[str],
    *,
    k: int,
    num_candidates: int,
    iterations: int,
    start_job_idx: int,
) -> tuple[list[float], int, int]:
    """Run a fixed number of timed queries for the measurement phase."""
    latencies: list[float] = []
    errors = 0
    job_idx = start_job_idx

    for _ in range(iterations):
        job_id = job_ids[job_idx % len(job_ids)]
        job_idx += 1
        elapsed_ms, ok = run_search(
            es,
            job_id,
            k=k,
            num_candidates=num_candidates,
        )
        if ok:
            latencies.append(elapsed_ms)
        else:
            errors += 1

    return latencies, errors, job_idx


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    index_config = VectorIndexConfig.from_env()
    es = get_es_client(ElasticsearchConfig.from_env())

    job_ids = load_job_ids(args.job_dataset)
    random.shuffle(job_ids)

    candidate_count = es.count(index=CANDIDATE_INDEX)['count']
    job_count = es.count(index=JOB_INDEX)['count']

    print('Search latency benchmark')
    print(f'  operation:     search_candidates_by_job_id')
    print(f'  candidate_index: {CANDIDATE_INDEX} ({candidate_count:,} docs)')
    print(f'  job_index:       {JOB_INDEX} ({job_count:,} docs)')
    print(f'  k:               {args.k}')
    print(f'  num_candidates:  {args.num_candidates}')
    print(f'  vector mapping:  {index_config.to_vector_field()}')
    print(f'  measurement:     {args.measurement_iterations:,} queries')

    # Warmup samples are excluded from the measurement summary below.
    _, warmup_errors, job_idx = warmup(
        es,
        job_ids,
        k=args.k,
        num_candidates=args.num_candidates,
        window_size=args.warmup_window_size,
        stable_windows=args.warmup_stable_windows,
        tolerance=args.warmup_tolerance,
        min_warmup=args.min_warmup,
        max_warmup=args.max_warmup,
    )

    if warmup_errors:
        print(f'  warmup errors: {warmup_errors:,}')

    # Continue job rotation from where warmup left off.
    measure_latencies, measure_errors, _ = measure(
        es,
        job_ids,
        k=args.k,
        num_candidates=args.num_candidates,
        iterations=args.measurement_iterations,
        start_job_idx=job_idx,
    )

    print_summary('Measurement', measure_latencies, measure_errors)


if __name__ == '__main__':
    main()
