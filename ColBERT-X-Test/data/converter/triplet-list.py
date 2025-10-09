import json

src = "ColBERT-X/data/triplet-fix.jsonl"
dst = "ColBERT-X/data/triplet-fixed-list.jsonl"

with open(src, "r") as fin, open(dst, "w") as fout:
    for line in fin:
        ex = json.loads(line)
        # Adjust keys depending on your file
        query = ex.get("query")
        pos = ex.get("positives")[0]
        neg = ex.get("negatives")[0]
        fout.write(json.dumps([query, pos, neg]) + "\n")

print("✅ Fixed file written to", dst)
