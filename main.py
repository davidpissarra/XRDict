from statistics import mode
import numpy as np
import argparse
import random
import torch
import json

from xrdict.data import XRDictDataset, device, data_processing
from xrdict.model import XRDict
from xrdict.train import train
from xrdict.vocab import Vocab

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--epoch_num', type=int, default=10)
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-sd', '--seed', type=int, default=999)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    setup_seed(args.seed)

    print('----------- Creating vocabulary -----------')
    vocab = Vocab('data/vec_inuse.json')

    print('----------- Creating model -----------')
    model = XRDict(ckpt_path='checkpoints/mlm_tlm_xnli15_1024.pth', vocab=vocab)

    bpe = json.load(open('./data/train_bpe.json', 'r', encoding='utf-8'))
    train_data, valid_data, test_data = data_processing(bpe, model.dico.word2id, vocab.word2id)

    train_dataset = XRDictDataset(train_data)
    valid_dataset = XRDictDataset(valid_data)
    test_dataset = XRDictDataset(test_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, args.epoch_num, (train_dataset, valid_dataset, train_dataset), optimizer, args.batch_size)
