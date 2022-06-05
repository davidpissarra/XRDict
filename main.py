from statistics import mode
import numpy as np
import argparse
import random
import torch
import json
import os

from xrdict.data import XRDictDataset, data_processing
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
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epoch_num', type=int, default=10)
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-sd', '--seed', type=int, default=999)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ep', '--embeds_path', type=str, default='data/vec_inuse.json')
    parser.add_argument('-dp', '--data_path', type=str, default='data/train_bpe15.json')
    parser.add_argument('-cp', '--xlmr_ckpt_path', type=str, default='checkpoints/mlm_tlm_xnli15_1024.pth')
    args = parser.parse_args()
    setup_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print('Loading data...')
    data = json.load(open(args.data_path, 'r', encoding='utf-8'))
    print(f'number of samples: {len(data)}')

    print('Creating vocabulary...')
    vocab = Vocab(args.embeds_path, set([sample['word'] for sample in data]))
    print(f'vocab size: {len(vocab.word2id)}')

    print('Creating XRDict model...')
    model = XRDict(xlmr_ckpt_path=args.xlmr_ckpt_path, vocab=vocab)

    print('Preprocessing data...')
    train_data, valid_data, test_data = data_processing(data, model.dico.word2id, vocab.word2id)
    print(f'train size: {len(train_data)}, valid size: {len(valid_data)}, test size: {len(test_data)}')

    train_dataset = XRDictDataset(train_data)
    valid_dataset = XRDictDataset(valid_data)
    test_dataset = XRDictDataset(test_data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, args.epoch_num, (train_dataset, valid_dataset, test_dataset), optimizer, args.batch_size, device)
