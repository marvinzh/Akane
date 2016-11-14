import string


def count_words(text):
    d = {}
    texts = text.split()
    for s in texts:
        try:
            d[s] += 1.
        except KeyError:
            d[s] = 1.
    return d


def remove_punctuation(text):
    text = str(text)

    ascii_index = list(map(lambda x: ord(x), string.punctuation))
    trans_list = {}
    for char in ascii_index:
        trans_list.setdefault(char)

    return text.translate(trans_list)


def gen_word_vector(text_data):
    pass


def dickeys2set(dic, t_set):
    keys = list(dic.keys())
    for key in keys:
        t_set.add(key)
    return t_set


def set2dic(t_set):
    dic = {}
    for item in t_set:
        dic[item] = 0.
    return dic



