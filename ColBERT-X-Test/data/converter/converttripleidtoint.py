import json

src = "ColBERT-X-Test/data/triplet-ready.jsonl"
dst = "ColBERT-X-Test/data/triplet-ready2.jsonl"

with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        q, pos, neg = json.loads(line)
        fout.write(json.dumps([int(q), int(pos), int(neg)]) + "\n")
