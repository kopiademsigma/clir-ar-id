import json

input_file = 'ColBERT-X-Test/data/triplet-fixed-list.jsonl'
output_file = 'ColBERT-X-Test/data/triplet-nway2.jsonl'

print("Converting triples to nway=2 format...")
print("=" * 60)

converted_count = 0
error_count = 0

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line_num, line in enumerate(fin, 1):
        try:
            triple = json.loads(line)
            
            # Handle list format: [query, pos_doc, neg_doc, ...]
            if isinstance(triple, list):
                if len(triple) < 3:
                    print(f"Warning: Line {line_num} has fewer than 3 elements, skipping")
                    error_count += 1
                    continue
                
                # Extract query (first element) and first positive and negative
                query = triple[0]
                positive = triple[1]
                negative = triple[2]
                
                # Create new structure with query as qid
                new_triple = {
                    "qid": f"q{line_num}",
                    "positives": [positive],
                    "negatives": [negative],
                    "nway": 2
                }
            
            # Handle dict format
            elif isinstance(triple, dict):
                # If already has the right structure, just update nway
                if 'qid' in triple and 'positives' in triple and 'negatives' in triple:
                    new_triple = {
                        "qid": triple['qid'],
                        "positives": triple['positives'][:1],  # Keep only first positive
                        "negatives": triple['negatives'][:1],  # Keep only first negative
                        "nway": 2
                    }
                else:
                    print(f"Warning: Line {line_num} has unexpected dict structure, skipping")
                    error_count += 1
                    continue
            
            else:
                print(f"Warning: Line {line_num} has unexpected type {type(triple)}, skipping")
                error_count += 1
                continue
            
            # Write the converted triple
            fout.write(json.dumps(new_triple, ensure_ascii=False) + '\n')
            converted_count += 1
            
            # Show progress every 1000 lines
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines...")
        
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON on line {line_num}: {e}")
            error_count += 1
            continue
        except Exception as e:
            print(f"Unexpected error on line {line_num}: {e}")
            error_count += 1
            continue

print("=" * 60)
print(f"\nConversion complete!")
print(f"Successfully converted: {converted_count} triples")
print(f"Errors encountered: {error_count}")
print(f"\nOutput saved to: {output_file}")

# Show first 3 examples
print("\n" + "=" * 60)
print("First 3 converted triples:")
print("=" * 60)
with open(output_file, 'r') as f:
    for i, line in enumerate(f, 1):
        if i > 3:
            break
        triple = json.loads(line)
        print(f"\nExample {i}:")
        print(json.dumps(triple, indent=2, ensure_ascii=False))

print("\n" + "=" * 60)
print("You can now use this file in your training command with --nway 2")
print("=" * 60)