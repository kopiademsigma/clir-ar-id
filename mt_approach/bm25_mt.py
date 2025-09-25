import numpy as np

from googletrans import Translator

import pandas as pd
from rank_bm25 import BM25Okapi

import re
from camel_tools.tokenizers.word import simple_word_tokenize  # you already installed camel-tools

def normalize_arabic(text):
    # remove tashkeel (diacritics), tatweel, normalize alef/ya/taa marbuta, remove non-Arabic extras
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)   # harakat
    text = re.sub(r'ـ', '', text)                             # tatweel
    text = re.sub(r'[إأآا]', 'ا', text)                        # alifs -> ا
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    # keep Arabic letters and spaces; replace other chars with space
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_arabic(text):
    text = normalize_arabic(str(text))
    try:
        toks = simple_word_tokenize(text)
    except Exception:
        toks = text.split()
    return toks


tsv_path = "../fath_muin/cleaned_corpus.tsv"
df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")

# Build token lists. If you saved 'tokens_space' use it; else tokenize from 'text'
if 'tokens_space' in df.columns:
    tokenized_corpus = [str(x).split() for x in df['tokens_space'].fillna('').tolist()]
else:
    tokenized_corpus = [tokenize_arabic(d) for d in df['text'].astype(str).tolist()]

# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# Keep docs (original text) handy for showing results
docs = df['text'].astype(str).tolist()


translator = Translator()

def translate_id_to_ar_google(query_id):
    out = translator.translate(query_id, src='id', dest='ar')
    return out.text


def retrieve_indonesian_query(query_id, topk=5, translator='marian'):
    # translate
    if translator == 'google':
        query_ar = translate_id_to_ar_google(query_id)
    # elif translator == 'marian':
    #     query_ar = translate_id_to_ar_marain(query_id)
    else:
        raise ValueError("translator must be 'google' or 'marian'")

    # normalize & tokenize (same pipeline as corpus)
    q_tokens = tokenize_arabic(query_ar)

    # get scores
    scores = bm25.get_scores(q_tokens)   # numpy array length = #docs

    # topk indices
    topk_idx = np.argsort(scores)[::-1][:topk]

    results = []
    for i in topk_idx:
        results.append({
            'rank': len(results) + 1,
            'line_id': int(df.iloc[i]['line_id']) if 'line_id' in df.columns else i,
            'score': float(scores[i]),
            'text': docs[i]
        })
    return query_ar, q_tokens, results

query_id = input()          # Indonesian input
query_ar, tokens, top_results = retrieve_indonesian_query(query_id, topk=10, translator='google')

print("Translated (AR):", query_ar)
print("Tokens:", tokens)
print("\nTop results:")
for r in top_results:
    print(r['rank'], "id:", r['line_id'], "score:", r['score'])
    print(r['text'][:300], "...")   # print a short snippet
    print("-" * 60)
