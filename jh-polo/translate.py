from rank_bm25 import BM25Okapi
from deep_translator import GoogleTranslator
import pandas as pd

# save with translation
translator = GoogleTranslator(source="ar", target="id")
with open("jh-polo/retrieved_passages.txt", "r", encoding="utf-8") as f:
    retrieved_text = f.readlines()
    for line in retrieved_text:
        if line.strip():
            translated = translator.translate(line, src="ar", dest="id")
with open("jh-polo/retrieved_passages_trans.txt", "w", encoding="utf-8") as g:
    for line in translated:
        g.write(line + "\n")

with open("jh-polo/relevant_passages.txt", "r", encoding="utf-8") as f:
    retrieved_text = f.readlines()
    for line in retrieved_text:
        if line.strip():
            translated = translator.translate(line, src="ar", dest="id")
with open("jh-polo/relevant_passages_trans.txt", "w", encoding="utf-8") as f:
    f.write(translated + "\n")

print("done")
