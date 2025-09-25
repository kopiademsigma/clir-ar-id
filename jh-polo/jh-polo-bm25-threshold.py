import pandas as pd
import random
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt_tab")

# Load chunked dataset
df = pd.read_csv("./fath_muin/chunks.csv")

# Tokenize each passage
tokenized_corpus = [word_tokenize(str(doc)) for doc in df["text"]]
bm25 = BM25Okapi(tokenized_corpus)

# Pick one random passage as "relevant"
random_idx = random.randint(0, len(df)-1)
relevant_passage = df.iloc[random_idx]["text"]
print("Relevant passage:\n", relevant_passage, "\n")

# Use the passage itself as query
query = word_tokenize(relevant_passage)

# Get scores for all passages
scores = bm25.get_scores(query)

# Exclude the query passage itself
scores[random_idx] = -1  

# Sort by score (descending)
sorted_indices = scores.argsort()[::-1]

# The top-scoring passage (not itself) becomes "positive"
pos_idx = sorted_indices[0]
pos_passage = df.iloc[pos_idx]["text"]
pos_score = scores[pos_idx]

# Now find a negative with ratio < 0.65
neg_idx, neg_passage, neg_score = None, None, None
for idx in sorted_indices[1:]:
    ratio = scores[idx] / pos_score
    if ratio < 0.65:
        neg_idx = idx
        neg_passage = df.iloc[idx]["text"]
        neg_score = scores[idx]
        break

# Write outputs
with open("jh-polo/relevant_passages.txt", "w", encoding="utf-8") as f:
    f.write(relevant_passage)

with open("jh-polo/retrieved_passages.txt", "w", encoding="utf-8") as f:
    f.write("Positive passage:\n" + pos_passage + "\n\n")
    if neg_passage:
        f.write("Negative passage (ratio < 0.65):\n" + neg_passage + "\n")

print("Positive passage:\n", pos_passage[:200], "...\n")
if neg_passage:
    print("Negative passage (ratio < 0.65):\n", neg_passage[:200], "...\n")
    print(f"Ratio = {neg_score/pos_score:.2f}")
else:
    print("⚠️ No suitable negative found with ratio < 0.65")
