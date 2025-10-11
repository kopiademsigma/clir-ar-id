import json

input_path = "ColBERT-X-Test/data/triplet-fixed-list.jsonl"
output_path = "ColBERT-X-Test/data/triplet-fixed-list.tsv"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        fout.write("\t".join(data) + "\n")

print("✅ Conversion done:", output_path)
