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

### Workflow

1. `notebooks/candidate_profile_embedding.ipynb` — embed candidates → index `candidate_profiles`
2. `notebooks/job_profile_embedding.ipynb` — embed jobs → index `job_profiles`
3. `notebooks/job_to_candidate_search.ipynb` — kNN search: job embedding → top-k candidates

Shared logic lives in `core/`. Both pipelines use `Qwen/Qwen3-Embedding-0.6B` (1024-dim cosine vectors).
