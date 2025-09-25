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
results = bm25.get_top_n(query, df["text"], n=5)

# Show retrieved passages (likely includes some irrelevant ones)
print("Retrieved passages:\n")

with open("jh-polo/relevant_passages.txt", "w", encoding="utf-8") as f:
    f.write(relevant_passage)

for i, passage in enumerate(results, 1):
    print(f"{i}. {passage[:200]}...\n")
    
print("Relevant passage:\n", relevant_passage, "\n")

with open("jh-polo/retrieved_passages.txt", "w", encoding="utf-8") as f:
    for i, passage in enumerate(results, 1):
        f.write(f"{i}. {passage}\n\n")

