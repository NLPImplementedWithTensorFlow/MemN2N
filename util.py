import numpy as np
import os

def mk_dict(data_paths):
    lines = []
    for data_path in data_paths:
        with open(data_path, "r") as fs:
            lines += fs.readlines()

    sentences = " ".join([line.split("\n")[0] for line in lines])
    words = list(set(sentences.split(" ")))
    return words

def read_dict(data_path):
    with open(data_path) as fs:
        lines = fs.readlines()

    lines = [line.split("\n")[0] for line in lines]
    return lines

def write_dict(data_path, dict_):
    with open(data_path, "a") as fs:
        fs.write("\n".join(dict_))
    return None

def convert_sentence2bow(sentence, dict_):
    bow = [0] * len(dict_)
    for word in sentence.split(" "):
        bow[dict_.index(word)] = 1
    return bow

def read_support_data(data_path):
    with open(data_path) as fs:
        sentences = fs.read().lower().replace(".", "")

    support_data = sentences.split("\n")
    return support_data

def read_q_label_data(data_path):
    with open(data_path) as fs:
        lines = fs.read().lower().replace("?", "").split("\n")

    q_ = []
    labels = []
    for line in lines:
        splitted = line.split("\t")
        q_.append(splitted[0])
        labels.append(splitted[1])

    return q_, labels


def mk_train_function(batch_size, support_path, q_label_path, dict_path):
    #q_label_path: question and answer of its data path
    #support path: support sentences data path

    dict_ = read_dict(dict_path)
    support_data = read_support_data(support_path)
    q_, label = read_q_label_data(q_label_path)

    def train_function():
        dump = []
        while True:

            yield

    return train_function
