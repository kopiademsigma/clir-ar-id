import json

print("Fixing triples to use query IDs instead of query text...")

# Load queries to create text-to-ID mapping
query_text_to_id = {}
with open('ColBERT-X-Test/data/queries.train.tsv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            qid = parts[0]
            query_text = parts[1]
            query_text_to_id[query_text] = qid

print(f"Loaded {len(query_text_to_id)} queries")

# Fix the triples file
input_file = 'ColBERT-X-Test/data/triplet-fixed-list2.jsonl'
output_file = 'ColBERT-X-Test/data/triplet-final.jsonl'

fixed_count = 0
missing_count = 0

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line_num, line in enumerate(fin, 1):
        triple = json.loads(line)
        
        if isinstance(triple, list) and len(triple) >= 3:
            query = triple[0]
            pos_id = triple[1]
            neg_id = triple[2]
            
            # Check if query is text and needs to be converted to ID
            if query in query_text_to_id:
                query_id = query_text_to_id[query]
                fixed_count += 1
            else:
                # Assume it's already an ID
                query_id = query
            
            new_triple = [query_id, pos_id, neg_id]
            fout.write(json.dumps(new_triple, ensure_ascii=False) + '\n')
        
        if line_num % 1000 == 0:
            print(f"Processed {line_num} lines...")

print(f"\nFixed {fixed_count} triples")
print(f"Output: {output_file}")

# Show first 3 examples
print("\nFirst 3 triples:")
with open(output_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i > 3:
            break
        print(f"{i}: {json.loads(line)}")