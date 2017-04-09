import numpy as np

WORD_EMBEDDING_FILE = "./lib/glove.6B.100d.txt"
WORD_EMBEDDING_SIZE = 100
MAX_SENTENCE_LENGTH = 60

word_map = None


def sentence2matrix(sentence):
    s = sentence.split()
    if len(s) > MAX_SENTENCE_LENGTH:
        started = (len(s)-MAX_SENTENCE_LENGTH)/2
        s = s[started:started + MAX_SENTENCE_LENGTH]

    ret = np.zeros((MAX_SENTENCE_LENGTH, WORD_EMBEDDING_SIZE))

    for i in range(len(s)):
        ret[i] = word2vec(s[i])

    return ret


def word2vec(word):
    if word_map is None:
        __load_word_map()

    if word in word_map:
        return word_map[word]
    else:
        return np.zeros(WORD_EMBEDDING_SIZE)


def __load_word_map():
    print("Load word2vec for the first time")
    global word_map
    word_map = {}

    # Load data from files
    word_data = list(open(WORD_EMBEDDING_FILE, "r").readlines())

    # Generate labels
    for word in word_data:
        t = word.split()
        word_map[t[0]] = np.array(list(map(float, t[1:])))

    print("Load word2vec finished")
