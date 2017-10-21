import numpy as np
import os
import random

def mk_dict(data_paths):
    lines = []
    for data_path in data_paths:
        with open(data_path, "r") as fs:
            content = fs.read().lower()
            if len(content.split("\t")) > 1:
                ##q_label_
                content = content.replace("?", "")
                content = " ".join([" ".join(line.split("\t")[:2]) for line in content.split("\n")])
            else:
                content = content.replace(".", "")

            lines += content.split("\n")

    sentences = " ".join(lines)
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
    support_data = [convert_sentence2bow(s_, dict_) for s_ in read_support_data(support_path)]
    q_, label = read_q_label_data(q_label_path)
    sets = np.array([[convert_sentence2bow(q_, dict_), convert_sentence2bow(label, dict_)] for q_, label in zip(q_, label)])

    def train_function():
        r_ = range(len(sets))
        while True:
            choiced_idx = [random.choice(r_) for _ in range(batch_size)]
            choiced = sets[choiced_idx]
            choiced_q = choiced[:, 0]
            choiced_label = choiced[:, 1]
            yield choiced_q, choiced_label
    return train_function, support_data
