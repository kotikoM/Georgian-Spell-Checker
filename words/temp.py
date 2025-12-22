def find_non_overlapping_repeating_substring(word):
    n = len(word)

    for length in range(2, n // 3 + 1):
        positions = {}

        for i in range(n - length + 1):
            sub = word[i:i + length]

            if sub in positions:
                positions[sub].append(i)

                # Check last three positions only (enough to decide)
                if len(positions[sub]) >= 3:
                    p1, p2, p3 = positions[sub][-3:]
                    if p2 - p1 >= length and p3 - p2 >= length:
                        return sub
            else:
                positions[sub] = [i]

    return ""


words = open('../words/ganmarteba.ge_words.txt', 'r', encoding='utf-8').read().split('\n')

for w in words:
    if w[:2] == 'რი':
        print(w)
    # sub = find_non_overlapping_repeating_substring(w)
    # if sub:
    #     print(w, '→', sub)
