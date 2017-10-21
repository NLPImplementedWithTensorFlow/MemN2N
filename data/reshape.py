import os

path = "tasks_1-20_v1-2/"
dirs = os.listdir(path)
files = [f for f in dirs if os.path.isdir(path+f)]

def write_data(path, content):
    with open(path, "a") as fs:
        fs.write("".join(content))

def mk_q_and_labels(lines):
    q_labels = []
    support_sentences = []
    for line in lines:
        line = " ".join(line.split(" ")[1:])
        if len(line.split("\t")) > 1:
            q_labels.append(line)
        else:
            support_sentences.append(line)
    return q_labels, support_sentences

def mk_data(dir_name, data_path):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    train_lines = []
    test_lines = []
    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name)) as fs:
            if len(file_name.split("train")) > 1:
                train_lines += fs.readlines()
            else:
                test_lines += fs.readlines()

    q_labels_train, support_sentences_train = mk_q_and_labels(train_lines)
    q_labels_test, support_sentences_test = mk_q_and_labels(test_lines)
    
    write_data(os.path.join(dir_name, "support_sentences_train.txt"), support_sentences_train)
    write_data(os.path.join(dir_name, "q_labels_train.txt"), q_labels_train)
    write_data(os.path.join(dir_name, "support_sentences_test.txt"), support_sentences_test)
    write_data(os.path.join(dir_name, "q_labels_test.txt"), q_labels_test)

[mk_data(path_, path+path_) for path_ in files]
