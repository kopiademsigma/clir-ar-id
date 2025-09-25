import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer resources
nltk.download("punkt_tab")

# Load dataset
df = pd.read_csv("./fath_muin/chunks.csv")

# Tokenize corpus
tokenized_corpus = [word_tokenize(str(doc)) for doc in df["text"]]
bm25 = BM25Okapi(tokenized_corpus)

pairs = []

# Iterate over every passage as the query
for q_idx, query_passage in enumerate(df["text"]):
    query = word_tokenize(str(query_passage))
    scores = bm25.get_scores(query)

    # Sort indices by score (descending), keep all (including itself)
    sorted_indices = scores.argsort()[::-1]

    # Positive is the query itself
    pos_passage = query_passage
    pos_score = scores[q_idx]

    # Find a negative with ratio < 0.65
    neg_idx, neg_passage, neg_score = None, None, None
    for idx in sorted_indices:
        if idx == q_idx:  # skip itself when searching negatives
            continue
        ratio = scores[idx] / pos_score if pos_score > 0 else 0
        if ratio < 0.65:
            neg_idx = idx
            neg_passage = df.iloc[idx]["text"]
            break

    if neg_passage:
        pairs.append({
            "positive_passage": pos_passage,
            "negative_passage": neg_passage
        })

# Save to CSV (tab-separated)
output_path = "jh-polo/passages_pairs.csv"
pd.DataFrame(pairs).to_csv(output_path, sep="\t", index=False)

print(f"✅ Saved {len(pairs)} passage pairs to {output_path}")
