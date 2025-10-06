import pandas as pd
import json
triples = []
with open("ColBERT-X/data/triplet-fix.jsonl") as f:
    for i, line in enumerate(f, start=1):
        item = json.loads(line)
        q = item["query"]
        for pos in item["positives"]:
            for neg in item["negatives"]:
                triples.append((i, pos, neg))

pd.DataFrame(triples).to_csv(
    "ColBERT-X/data/triples.train.tsv", sep="\t", index=False, header=False
)
