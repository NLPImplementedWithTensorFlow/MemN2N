from model import *
import argparse
import os
from util import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--itr", dest="itr", type=int, default=10000)
    parser.add_argument("--lr", dest="lr", type=float, default= 0.2) 
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=5)
    parser.add_argument("--layer_num", dest="layer_num", type=int, default=3)
    parser.add_argument("--support_size", dest="support_size", type=int, default=10)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int, default=921)
    parser.add_argument("--embedding_size", dest="embedding_size", type=int, default=32)
    parser.add_argument("--support_path", dest="support_path", type=str, default="data/en/support_sentences_train.txt")
    parser.add_argument("--q_label_path", dest="q_label_path", type=str, default="data/en/q_labels_train.txt")
    parser.add_argument("--dict_path", dest="dict_path", type=str, default="data/dict.txt")
    parser.add_argument("--train", dest="train", type=bool, default=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dict_path):
        words = mk_dict([args.q_label_path, args.support_path])
        write_dict(args.dict_path, words)

    if not os.path.exists("saved"):
        os.mkdir("saved")

    model_ = model(args)
    if args.train:
        model_.train()
