# semantic-search

Match job descriptions to candidate profiles via Qwen embeddings and Elasticsearch kNN.

### Setup

```bash
git clone git@github.com:zachmchuang/semantic-search.git && cd semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && mkdir data
```

Download parquet files into `data/`:

- Candidates: [lang-uk/recruitment-dataset-candidate-profiles-english](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) → `candidate_profiles.parquet`
- Jobs: [lang-uk/recruitment-dataset-job-descriptions-english](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) → `job_profiles.parquet`

`.env`:

```
EMBED_API_KEY=...
ES_PASSWORD=...
```

Optional Elasticsearch vector settings (defaults shown in `core/config.py` → `VectorIndexConfig`):

```
ES_VECTOR_INDEX_OPTIONS_TYPE=int8_hnsw
ES_VECTOR_INDEX_OPTIONS_M=16
ES_VECTOR_INDEX_OPTIONS_EF_CONSTRUCTION=100
```

### Workflow

1. `notebooks/candidate_profile_embedding.ipynb` — embed candidates → index `candidate_profiles`
2. `notebooks/job_profile_embedding.ipynb` — embed jobs → index `job_profiles`
3. `notebooks/job_to_candidate_search.ipynb` — kNN search: job embedding → top-k candidates

Shared logic lives in `core/`. Both pipelines use `Qwen/Qwen3-Embedding-0.6B` (1024-dim cosine vectors).

### Index configuration

Vector index settings (`dense_vector` mapping, HNSW `index_options`, shards/replicas) are centralized in `VectorIndexConfig` (`core/config.py`). Edit defaults there or override via env vars. Both embedding notebooks pass `index_config` to `create_index()` and `candidate_mappings()` / `job_mappings()`.

Current defaults: `int8_hnsw`, `m=16`, `ef_construction=100`, `similarity=cosine`, `dims=1024`.

**Reindexing after mapping changes:** mapping is fixed at index creation. Re-run the embedding notebooks with existing `.npy` memmaps (skip `embed_all`) — `create_index(..., recreate=True)` deletes and recreates the index, then `bulk_index()` reloads documents.

### Search

- `search_candidates_by_job_id()` — look up a stored job vector in `job_profiles`, kNN against `candidate_profiles` (no embedding API at query time; requires ES 9.4+ lookup API)
- `search_candidates_by_profile_text()` — embed ad-hoc text, then kNN search

Default query params: `k=10`, `num_candidates=100`.

### Benchmarks

```bash
# Search latency baseline (job → candidates, warmup + 1000 measured queries)
python scripts/benchmark_search_latency.py

# Embedding API throughput
python scripts/benchmark_batch_sizes.py
python scripts/benchmark_concurrency.py
```

`benchmark_search_latency.py` runs `search_candidates_by_job_id` with adaptive warmup (stabilizes on cumulative running p50), then reports p50/p90/p95/p99 and throughput. Tune via `--k`, `--num-candidates`, `--warmup-tolerance`, etc.
