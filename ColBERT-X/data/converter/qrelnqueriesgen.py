import pandas as pd

pd.read_csv("ColBERT-X/data/eval-set.csv")[["query_id","query"]].to_csv(
    "ColBERT-X/data/queries.dev.tsv", sep="\t", index=False, header=False
)
pd.read_csv("ColBERT-X/data/eval-set.csv")[["query_id","relevant_passage_ids"]].assign(
    zero=0, relevance=1
)[["query_id","zero","relevant_passage_ids","relevance"]].to_csv(
    "ColBERT-X/data/qrels.dev.tsv", sep="\t", index=False, header=False
)
