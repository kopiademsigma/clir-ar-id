import pandas as pd

df = pd.read_csv("ColBERT-X/data/chunks.csv")
df[["chunk_id", "text"]].to_csv("ColBERT-X/data/collection.tsv", sep="\t", index=False, header=False)
