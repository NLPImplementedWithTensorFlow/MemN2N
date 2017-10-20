import os

path = "tasks_1-20_v1-2/"
dirs = os.listdir(path)
files = [f for f in dirs if os.path.isdir(path+f)]

def write_data(path, content):
    with open(path, "a") as fs:
        fs.write("".join(content))

def mk_data(dir_name, data_path):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    lines = []
    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name)) as fs:
            lines += fs.readlines()

    q_labels = []
    support_sentences = []
    for line in lines:
        line = " ".join(line.split(" ")[1:])
        if len(line.split("\t")) > 1:
            q_labels.append(line)
        else:
            support_sentences.append(line)
    
    write_data(os.path.join(dir_name, "support_sentence.txt"), support_sentences)
    write_data(os.path.join(dir_name, "q_labels.txt"), q_labels)

[mk_data(path_, path+path_) for path_ in files]
