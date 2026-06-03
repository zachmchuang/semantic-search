# semantic-search

### Setup
```
git clone git@github.com:zachmchuang/semantic-search.git
cd semantic-search
mkdir data
```
Download [this dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english)

`mv train-00000-of-00001.parquet data/candidates.parquet`

### Project Description

We have a dataset of candidate profiles. Convert each candidate into a profile string. Embed each profile string with [https://huggingface.co/microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) model using the 'semantically similar text' prompt. Load all of the embedded profiles into ElasticSearch/Qdrant seperately and run vector search on embedded job descriptions of the same format. 
