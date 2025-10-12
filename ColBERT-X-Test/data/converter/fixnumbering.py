# Fix collection.tsv and triples to have 0-indexed sequential IDs
# Step 1: Create a mapping from old IDs to new IDs
print("Step 1: Creating ID mapping...")
old_to_new_id = {}
with open('ColBERT-X-Test/data/collection.tsv', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            old_id = parts[0]
            old_to_new_id[old_id] = str(idx)

print(f"Created mapping for {len(old_to_new_id)} documents")

# Step 2: Fix collection.tsv
print("\nStep 2: Fixing collection.tsv...")
with open('ColBERT-X-Test/data/collection.tsv', 'r', encoding='utf-8') as fin:
    with open('ColBERT-X-Test/data/collection-fixed.tsv', 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(fin):
            parts = line.strip().split('\t', 1)
            if len(parts) >= 2:
                doc_text = parts[1]
                fout.write(f"{idx}\t{doc_text}\n")

# Step 3: Fix triples file
print("\nStep 3: Fixing triples file...")
import json
with open('ColBERT-X-Test/data/triplet-fixed-list.jsonl', 'r', encoding='utf-8') as fin:
    with open('ColBERT-X-Test/data/triplet-fixed-list2.jsonl', 'w', encoding='utf-8') as fout:
        for line in fin:
            triple = json.loads(line)
            if isinstance(triple, list) and len(triple) >= 3:
                qid = triple[0]
                pos_id = triple[1]
                neg_id = triple[2]
                
                # Map old IDs to new IDs
                new_pos_id = old_to_new_id.get(pos_id, pos_id)
                new_neg_id = old_to_new_id.get(neg_id, neg_id)
                
                new_triple = [qid, new_pos_id, new_neg_id]
                fout.write(json.dumps(new_triple, ensure_ascii=False) + '\n')

print("\nDone! Use these files:")
print("  --training_collection ColBERT-X-Test/data/collection-fixed.tsv")
print("  --training_triples ColBERT-X-Test/data/triplet-fixed-list2.jsonl")