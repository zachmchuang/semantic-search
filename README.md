# semantic-search

### Setup
```
git clone git@github.com:zachmchuang/semantic-search.git
cd semantic-search
mkdir data
```
Download [this dataset](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) of candidate profiles. 

`mv train-00000-of-00001.parquet data/candidates.parquet`

### Project Description

We have a dataset of candidate profiles. Convert each candidate into a profile string. Embed each profile string. Load all of the embedded profiles into ElasticSearch and run benchmark tests. 
