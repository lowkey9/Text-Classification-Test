import csv
import numpy as np
import random
from gensim.models import Word2Vec
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence

wvPath = '.\\Data\\word2vec.bin'
numClasses = 14
maxLen = 350

def load_data(data_only):
    # num == 1, get data only; type == 0, get data and label
    file = '.\\Data\\shuffled-full-set-hashed.csv'
    with open(file) as infile:
        content = csv.reader(infile, delimiter=',')
        label = []
        data = []
        for row in content:
            curLabel = row[0]
            curData = row[1].split()
            label.append(curLabel)
            data.append(curData)

    if data_only == 1:
        return data
    else:
        return data, label


def get_dictionary(data):
    dic = dict()
    i = 0
    for d in data:
        for word in d:
            if word not in dic:
                dic[word] = i
                i += 1
    return dic


def init_embed(dic):
	# initialize embedding matrix
    wv = Word2Vec.load(wvPath)
    
    embedding_matrix = np.zeros((len(dic) + 1, 100))
    for word, i in dic.items():
        embedding_vector = wv[str(word)]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            # words not found in dictionary will be all-zeros
    return embedding_matrix


def embed_onehot(doc, dic):
    # embed text input using index of words
    text_data = []
    for d in doc:
        text = []
        for word in d:
            index = 0
            if word in dic:
                index = dic[word]
            text.append(index)
        text_data.append(text)
    return text_data


def get_text_data(data, dic):
    res = []
    for doc in data:
        text = embed_onehot(doc, dic)
        res.append(text)
    return res


def seperate_data(data, label):
	# 8 : 2 train-test split, but use k-fold later
    train_data, test_data = [], []
    train_label, test_label = [], []
    i = 0
    while i < len(data):
        if random.randint(0, 100) < 80:
            train_data.append(data[i])
            train_label.append(label[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
        i += 1
    return train_data, train_label, test_data, test_label


def get_data():
    data, label = load_data(0)
    dic = get_dictionary(data)
    embed_matrix = init_embed(dic)
    text_data = get_text_data(data, dic)
    # cat_label, mapping = categorical(label, numClasses)
    unique_label, cat_label = np.unique(label, return_inverse=True)
    cat_label = to_categorical(cat_label, numClasses)
    train_data, train_label, test_data, test_label = seperate_data(text_data, cat_label)
    train_data = sequence.pad_sequences(train_data, maxLen)
    test_data = sequence.pad_sequences(test_data, maxLen)
    return train_data, train_label, test_data, test_label, embed_matrix, dic, unique_label

def get_onehot_data():
    data, label = load_data(0)
    dic = get_dictionary(data)
    embed_matrix = init_embed(dic)
    text_data = embed_onehot(data, dic)     # raw one-hot
    unique_label, cat_label = np.unique(label, return_inverse=True)
    cat_label = to_categorical(cat_label, numClasses)
    text_data = sequence.pad_sequences(text_data, maxLen)
    return text_data, cat_label, dic, unique_label
