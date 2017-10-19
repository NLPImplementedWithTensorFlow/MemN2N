import os

path = "tasks_1-20_v1-2/"
dirs = os.listdir(path)
files = [f for f in dirs if os.path.isdir(path+f)]

def mk_data(dir_name, data_path):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



[mk_data(path_, path+path_) for path_ in files]
