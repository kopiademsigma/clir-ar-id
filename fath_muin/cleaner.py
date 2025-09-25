import re

def clean_openiti_text(text: str) -> str:
    # Remove OpenITI headers and metadata
    text = re.sub(r'^######OpenITI#.*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#META#.*\n?', '', text, flags=re.MULTILINE)

    # Remove page markers (both inline and standalone)
    text = re.sub(r'PageV\d+P\d+', '', text)

    # Remove span-like tags
    text = re.sub(r'</?span.*?>', '', text)

    # Remove artifacts (~~, ## |, leading #)
    text = text.replace('~~', ' ')
    text = re.sub(r'##\s*\|', '', text)
    text = re.sub(r'^#\s*', '', text, flags=re.MULTILINE)

    # Remove Latin words (keep Arabic only)
    text = re.sub(r'\b[A-Za-z0-9]+\b', '', text)

    # Normalize punctuation spacing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # preserve paragraph breaks

    return text.strip()
# === Main script ===
input_file = "0987ZaynDinMalibari.FathMucin.Shamela0011327-ara1.txt"
output_file = "0987ZaynDinMalibari.FathMucin.cleaned.txt"

# Read raw text
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Clean the text
cleaned_text = clean_openiti_text(raw_text)

# Save cleaned text
with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"✅ Cleaned file saved as {output_file}")