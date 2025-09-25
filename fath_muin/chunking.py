import csv

def chunk_text(words, chunk_size=300):
    """Split list of words into fixed-length chunks."""
    for i in range(0, len(words), chunk_size):
        yield words[i:i+chunk_size]

def prepare_chunks(input_file, output_file, doc_id, chunk_size=300):
    # Read cleaned text
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Tokenize by whitespace
    words = text.split()
    
    # Create chunks
    chunks = list(chunk_text(words, chunk_size))
    
    # Write chunks into CSV
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["doc_id", "chunk_id", "text"])
        writer.writeheader()
        
        for idx, chunk in enumerate(chunks, start=1):
            writer.writerow({
                "doc_id": doc_id,
                "chunk_id": idx,
                "text": " ".join(chunk)
            })

# Example usage
prepare_chunks(
    input_file="0987ZaynDinMalibari.FathMucin.cleaned.txt",
    output_file="chunks.csv",
    doc_id="0987ZaynDinMalibari",
    chunk_size=300  # <-- adjust depending on IR model
)
