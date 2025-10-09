import json, pandas as pd

queries = []
with open("ColBERT-X/data/triplet-fix.jsonl") as f:
    for i, line in enumerate(f, start=1):
        item = json.loads(line)
        queries.append((i, item["query"]))

pd.DataFrame(queries, columns=["query_id", "query"]).to_csv(
    "ColBERT-X/data/queries.train.tsv", sep="\t", index=False, header=False
)
