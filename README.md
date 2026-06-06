# semantic-search

Semantic search for matching job descriptions to candidate profiles using Qwen embeddings and Elasticsearch kNN.

### Setup

```
git clone git@github.com:zachmchuang/semantic-search.git
cd semantic-search
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir data
```

Download datasets from Hugging Face:

- [Candidate profiles (English)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)
- [Job descriptions (English)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english)

```
mv train-00000-of-00001.parquet data/candidate_profiles.parquet
mv train-00000-of-00001.parquet data/job_profiles.parquet
```

Create a `.env` file in the project root:

```
EMBED_API_KEY=your_key
ES_PASSWORD=your_elasticsearch_password
# optional:
# ES_URL=https://localhost:9200
# ES_USER=elastic
```

### Project structure

- `core/` — shared Python module (embedding, indexing, search)
- `notebooks/candidate_profile_embedding.ipynb` — embed and index candidates
- `notebooks/job_profile_embedding.ipynb` — embed and index jobs
- `notebooks/job_to_candidate_search.ipynb` — kNN search: job → candidates

### Workflow

1. **Index candidates** — run `candidate_profile_embedding.ipynb`. Creates Elasticsearch index `candidate_profiles` with 1024-dim cosine vectors.

2. **Index jobs** — run `job_profile_embedding.ipynb`. Creates index `job_profiles`. Embeddings checkpoint to `data/job_embeddings.npy` with resume support via `data/job_embed_progress.json`.

3. **Search** — run `job_to_candidate_search.ipynb`. Uses a job's stored embedding to find top-k matching candidates in `candidate_profiles`.

Both embedding pipelines use the same model (`Qwen/Qwen3-Embedding-0.6B`) so job and candidate vectors are comparable.

### Elasticsearch indices

| Index | Contents |
|-------|----------|
| `candidate_profiles` | Candidate profile text, metadata, embeddings |
| `job_profiles` | Job profile text, metadata, embeddings |

Search direction: job embedding is the query vector; `candidate_profiles.embedding` is the search corpus.
