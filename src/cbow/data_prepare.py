import collections
import jieba_fast as cjieba

def read_data_ancient(file_name):
    """
    read poem and do word segmentation

    Returns
    -------
    data : list
        list of word segmentation in sequence order
    """
    data = list()
    with open(file_name, "r") as fp:
        while True:
            line = fp.readline()
            if len(line) == 0:
                break
            line = line.strip()
            for ch in line:
                data.append(ch)
    return data

def read_data_modern(file_name):
    data = list()
    with open(file_name, "r") as fp:
        while True:
            line = fp.readline()
            if len(line) == 0:
                break
            line = line.strip()
            word_list = line.split(",")
            data.extend(word_list)
    return data

def read_data(file_name, mode="ancient"):
    if mode=="ancient":
        return read_data_ancient(file_name)
    else:
        return read_data_modern(file_name)

def build_dataset(words, vocabulary_size):
    """
    build dataset to be used for training

    Returns
    -------
    data : list
        list of words mapped to corresponding index

    count: list of tuple
        A list of (word, word_count) tuples ordered by word count desc

    dictionary: dict
        A mapping of word to embeddings row indices.

    reverse_dictionary: dict
        A mapping of embeddings row indices to word
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
