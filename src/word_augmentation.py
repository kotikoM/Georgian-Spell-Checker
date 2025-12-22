import random

GEORGIAN_LETTERS = set("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")
GEORGIAN_QWERTY_KEYBOARD_NEIGHBORS = {
    "ა": "ქწსზ",
    "ბ": "ვგჰნ",
    "გ": "ფტჰბვ",
    "დ": "ერფცხს",
    "ე": "წსდრ",
    "ვ": "ცფგბ",
    "ზ": "ასხ",
    "თ": "ღრფგყ",
    "ი": "უჯკო",
    "კ": "იოლმჯ",
    "ლ": "კოპ",
    "მ": "ნჯკლ",
    "ნ": "ბჰჯმ",
    "ო": "იკლპ",
    "პ": "ოლ",
    "ჟ": "ჰუიკმნ",
    "რ": "ედფგტ",
    "ს": "აწდხზ",
    "ტ": "რფგჰყ",
    "უ": "ყჰჯკი",
    "ფ": "დრტგვც",
    "ქ": "წსა",
    "ღ": "ედფგტ",
    "ყ": "ტგჰჯუ",
    "შ": "აჭწდძზხ",
    "ჩ": "ხდფვ",
    "ც": "ხდფვ",
    "ძ": "შასხ",
    "წ": "ქასდე",
    "ჭ": "ქაშსდე",
    "ხ": "ზსდც",
    "ჯ": "ჰუიკმნ",
    "ჰ": "გყუჯნბ",
}


def delete_char(word):
    if len(word) <= 1:
        return word
    i = random.randrange(len(word))
    return word[:i] + word[i + 1:]


def insert_random_char(word):
    i = random.randrange(len(word) + 1)
    char = random.choice(tuple(GEORGIAN_LETTERS))
    return word[:i] + char + word[i:]


def duplicate_char(word):
    i = random.randrange(len(word))
    return word[:i] + word[i] + word[i:]


def substitute_char(word):
    i = random.randrange(len(word))
    char = word[i]
    neighbors = GEORGIAN_QWERTY_KEYBOARD_NEIGHBORS.get(char)
    if not neighbors:
        return word
    replacement = random.choice(neighbors)
    return word[:i] + replacement + word[i + 1:]


def swap_chars(word):
    if len(word) < 2:
        return word
    i = random.randrange(len(word) - 1)
    return (
            word[:i]
            + word[i + 1]
            + word[i]
            + word[i + 2:]
    )


ERROR_FUNCTIONS = [
    delete_char,
    duplicate_char,
    substitute_char,
    swap_chars,
]

def get_corrupted_words(word, number_of_corrupted_words=2):
    """"
    Character-Level Synthetic Error Generation
    """

    corrupted_words = set()

    word_len = len(word)

    # for word len < 10 1-2 error
    # for word len > 10 3-5 error
    word_error_count_threshold = 10
    if word_len < word_error_count_threshold:
        min_error, max_error = 1, 1
    else:
        min_error, max_error = 2, 3

    while len(corrupted_words) <= number_of_corrupted_words:
        corrupted = word
        num_errors = random.randint(min_error, max_error)

        # apply error functions
        for _ in range(num_errors):
            error_fn = random.choice(ERROR_FUNCTIONS)
            corrupted = error_fn(corrupted)

        corrupted_words.add(corrupted)

    return list(corrupted_words)


if __name__ == "__main__":
    words = ['თბილისი', 'ქუთაისი', 'ბათუმი', 'თელავი']

    print("\n__ word augmentation examples__")
    for w in words:
        print(f"\n Word: {w}")
        for i, c in enumerate(get_corrupted_words(w), 1):
            print(f"   {i:>2}. {c}")
