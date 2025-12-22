from collections import defaultdict
import pdfplumber
import re

PDF_PATH = "Didostati_Konstantines_Marjvena.pdf"

GEORGIAN_WORDS_REGEX = re.compile(r"[ა-ჰ]+(-[ა-ჰ]+)?")
UNWANTED_PUNCTUATION = ".,!?;:\"()[]{}“”"

# word -> number_of_appearance
word_count = defaultdict(int)

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"Processing page {i}")
        text = page.extract_text()
        if not text:
            continue

        for token in text.split():
            token = token.strip(UNWANTED_PUNCTUATION)
            if GEORGIAN_WORDS_REGEX.fullmatch(token):
                word_count[token] += 1

print(f"Extracted {len(word_count.keys())} unique Georgian words")
for k, v in sorted(word_count.items(), key=lambda item: item[1], reverse=True):
    print(f"word {k} appeared {v}")

word_txt_file = PDF_PATH.replace('.pdf', '_words.txt')
with open(word_txt_file, 'w', encoding="utf-8") as f:
    for w in sorted(word_count.keys()):
        f.write(w + "\n")
