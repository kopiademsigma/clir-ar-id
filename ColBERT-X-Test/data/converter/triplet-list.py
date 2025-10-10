import json

src = "ColBERT-X-Test/data/triplet-fix.jsonl"
dst = "ColBERT-X-Test/data/triplet-fixed-list-nway2.jsonl"

with open(src, "r") as fin, open(dst, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        # Adjust keys depending on your file
        query = ex.get("query")
        pos = ex.get("positives")[0]
        neg = ex.get("negatives")[0]
        fout.write(json.dumps([query, pos, neg, 2]) + "\n")

print("✅ Fixed file written to", dst)
