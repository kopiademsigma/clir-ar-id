# Define the names of your input and output files
file_a = 'jh-polo/query_passage_triples-all.tsv'
file_b = 'jh-polo/query_passage_triples-all-2.tsv'
output_file = 'jh-polo/query_passage_triples-5-fixed.tsv'

# Define how many lines to take from each file
lines_from_a = 1235 # Replace 10 with the number of lines you want from file A
lines_from_b = 595   # Replace 5 with the number of lines you want from file B

# --- Script starts here ---

try:
    # Open and read the specified number of lines from file A
    with open(file_a, 'r') as f_a:
        content_a = f_a.readlines()
        content_a = content_a[:lines_from_a]

    # Open and read the specified number of lines from file B
    with open(file_b, 'r') as f_b:
        content_b = f_b.readlines()
        content_b = content_b[:lines_from_b]

    # Combine the content from both files
    combined_content = content_a + content_b

    # Write the combined content to the output file
    with open(output_file, 'w') as f_out:
        f_out.writelines(combined_content)

    print(f"Successfully combined {len(content_a)} lines from '{file_a}' and {len(content_b)} lines from '{file_b}' into '{output_file}'.")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the input files are in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")