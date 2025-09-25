# bm25_clir_app.py
import pandas as pd
from rank_bm25 import BM25Okapi
from googletrans import Translator
import streamlit as st

# --- Load corpus once ---
@st.cache_resource
def load_bm25():
    tsv_path = "../fath_muin/cleaned_corpus.tsv"
    df = pd.read_csv(tsv_path, sep="\t").dropna(subset=["text"])
    tokenized_corpus = [str(text).split() for text in df["text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return df, bm25

df, bm25 = load_bm25()
translator = Translator()

# --- UI ---
st.title("🔎 Bilingual Search: Indonesian → Arabic (BM25)")
st.write("Type your Indonesian query below. It will be translated to Arabic and used to search the corpus.")

query_id = st.text_input("Indonesian query", "")

if query_id:
    # Translate Indonesian → Arabic
    translated_query = translator.translate(query_id, src="id", dest="ar").text
    st.markdown(f"**Translated query (Arabic):** {translated_query}")

    # Run BM25
    tokenized_query = translated_query.split()
    scores = bm25.get_scores(tokenized_query)
    df["score"] = scores
    top_results = df.sort_values("score", ascending=False).head(5)

    st.subheader("Top 5 Results")
    for idx, row in top_results.iterrows():
        st.markdown(f"""
        **Score {row['score']:.4f}**  
        
        **Index : {idx}** 

        **Snippet (Arabic):** {row['text']}...
        """)

