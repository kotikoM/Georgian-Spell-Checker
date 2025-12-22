import requests
import time
from bs4 import BeautifulSoup
import re


def extract_words(html):
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("ul.container.threecolumns li a")
    words_per_request = []

    for a in anchors:
        for sup in a.find_all("sup"):
            sup.decompose()

        word = a.get_text(strip=True)

        if re.fullmatch(r"[ა-ჰ]+(-[ა-ჰ]+)?", word):
            words_per_request.append(word)

    return words_per_request


def get_max_page(letter, upper_limit=2000):
    overshoot_page = upper_limit
    url = BASE_URL.format(letter=letter, page=overshoot_page)
    resp = requests.get(url, timeout=30)

    soup = BeautifulSoup(resp.text, "html.parser")

    meta = soup.find("meta", property="og:url")
    if meta and "content" in meta.attrs:
        match = re.search(r'/(\d+)$', meta["content"])
        if match:
            return int(match.group(1))

    return 1


BASE_URL = "https://www.ganmarteba.ge/search/{letter}/{page}"
alphabet = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
words = set()

for letter in alphabet:
    start_time = time.time()
    max_page = get_max_page(letter)
    for page in range(1, max_page + 1):
        url = BASE_URL.format(letter=letter, page=page)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Request failed for {url}: {e}")

            page += 1
            continue

        words_per_request = extract_words(resp.text)
        if len(words_per_request) < 30:
            print(letter, page, words_per_request)

        words.update(words_per_request)
        print(f"Letter {letter}  page {page}: +{len(words_per_request)} words")

        page += 1
        time.sleep(0.3)  # polite delay

    end_time = time.time()
    print(f"Letter {letter} took {(end_time - start_time):.2f} seconds")

print(f"Total words: {len(words)}")

with open("ganmarteba_words.txt", "w", encoding="utf-8") as f:
    for w in sorted(words):
        f.write(w + "\n")
