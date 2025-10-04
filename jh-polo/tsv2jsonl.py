import json

input_file = "jh-polo/query_passage_triples-5-fixed.tsv"
output_file = "jh-polo/triplet-fix.jsonl"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue  # skip baris yang salah format
        q, pos, neg = parts
        record = {
            "query": q,
            "positives": [pos],   # wrap in list
            "negatives": [neg]    # wrap in list
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
