import json

print("Creating properly mapped triples...")

# Step 1: Load document text to ID mapping from collection-fixed.tsv
print("\nStep 1: Loading collection...")
doc_text_to_id = {}
with open('ColBERT-X-Test/data/collection-fixed.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) >= 2:
            doc_id = parts[0]
            doc_text = parts[1]
            doc_text_to_id[doc_text] = doc_id

print(f"Loaded {len(doc_text_to_id)} documents")

# Step 2: Fix triples by mapping document texts to IDs
print("\nStep 2: Mapping triples...")
input_file = 'ColBERT-X-Test/data/triplet-final.jsonl'
output_file = 'ColBERT-X-Test/data/triplet-ready.jsonl'

mapped_count = 0
missing_pos = 0
missing_neg = 0

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line_num, line in enumerate(fin, 1):
        triple = json.loads(line)
        
        if isinstance(triple, list) and len(triple) >= 3:
            qid = triple[0]
            pos_text = triple[1]
            neg_text = triple[2]
            
            # Map document texts to IDs
            pos_id = doc_text_to_id.get(pos_text)
            neg_id = doc_text_to_id.get(neg_text)
            
            if pos_id is None:
                missing_pos += 1
                print(f"Warning: Positive doc not found for query {qid}, skipping triple")
                continue
            
            if neg_id is None:
                missing_neg += 1
                print(f"Warning: Negative doc not found for query {qid}, skipping triple")
                continue
            
            new_triple = [qid, pos_id, neg_id]
            fout.write(json.dumps(new_triple, ensure_ascii=False) + '\n')
            mapped_count += 1
        
        if line_num % 500 == 0:
            print(f"Processed {line_num} lines...")

print(f"\nSuccessfully mapped {mapped_count} triples")
print(f"Missing positive docs: {missing_pos}")
print(f"Missing negative docs: {missing_neg}")
print(f"\nOutput: {output_file}")

# Verify
print("\nFirst 5 mapped triples:")
with open(output_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i > 5:
            break
        print(f"  {i}: {json.loads(line)}")