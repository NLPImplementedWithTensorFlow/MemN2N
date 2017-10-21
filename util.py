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

def mk_train_function(batch_size, data_path, dict_path):
    dict_ = read_dict(dict_path)

    def train_function():
        while True:

            yield

    return train_function
