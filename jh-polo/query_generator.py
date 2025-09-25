import google.generativeai as genai

# Ganti 'YOUR_API_KEY' dengan kunci API Anda
genai.configure(api_key='AIzaSyD7Rg-QUVXCyzi03PFCzT1BxcbNMhstqjI')

# Pilih model yang ingin Anda gunakan, misalnya 'gemini-1.5-flash'
model = genai.GenerativeModel('gemini-2.5-flash')


def generate_and_write_query(input_file, output_file):
    """
    Membaca passage dari file, meng-generate query, dan menulis hasilnya ke file lain.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        # Asumsi format file input Anda: <passage_pos> \t <passage_neg> per baris
        # Melewati baris pertama (header)
        next(f_in) 
        for line_num, line in enumerate(f_in, start=1):
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            passage_pos = parts[0]
            passage_neg = parts[1]

            # Bentuk prompt dengan data dari file
            # prompt = f"""
            # Buat satu pertanyaan pencarian dalam bahasa Indonesia yang secara spesifik hanya berkaitan dengan informasi di dalam passage A. Pertanyaan ini harus tidak relevan dengan informasi yang ada di dalam passage B.

            # Passage A (positif):
            # {passage_pos}

            # Passage B (negatif):
            # {passage_neg}

            # Tuliskan outputnya dengan format berikut:
            # <pertanyaan_pencarian> \t {passage_pos} \t {passage_neg}
            # """
            prompt = f"""
            I will give you two Arabic passages. 
            - Passage A is the relevant one. 
            - Passage B is the non-relevant one.  

            Your task: generate 5 natural search queries in **Indonesian** such that 
            a user asking these queries would find Passage A relevant, 
            and Passage B irrelevant.  

            Rules:
            - Queries should be short and look like what a normal person would type in a search engine.
            - Do not refer to “Passage A” or “Passage B”.
            - Do not mention that you are comparing passages.
            - Write only the queries, one per line, without numbering.

            Passage A:
            {passage_pos}

            Passage B:
            {passage_neg}
            """

            try:
                # Kirim prompt ke Gemini
                response = model.generate_content(prompt)
                
                # Pastikan respons tidak kosong
                if response.text:
                   # Split into lines to handle multiple queries
                    queries = response.text.strip().split("\n")
                    for q in queries:
                        q = q.strip()
                        if q:  # avoid empty lines
                            # Force correct tab-separated format
                            f_out.write(f"{q}\t{passage_pos}\t{passage_neg}\n")
                    print(f"✅ Line {line_num}: Queries successfully generated and written.")
            except Exception as e:
                print(f"⚠️ Line {line_num}: Failed to generate query → {e}")
                # Jika gagal, tulis baris kosong atau pesan error
                f_out.write(f"ERROR \t {passage_pos} \t {passage_neg}\n")

# Gunakan fungsi di atas
input_filename = "jh-polo/passages_pairs.csv"  # Ganti dengan nama file Anda
output_filename = "jh-polo/query_passage_triples.csv"

# Jalankan skrip
generate_and_write_query(input_filename, output_filename)