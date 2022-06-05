import json
import torch
import os, gc
import random
import numpy as np
import torch.utils.data
from xlm.data.dictionary import UNK_WORD


class XRDictDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]

    @staticmethod
    def collate_fn(batch):
        bs = len(batch)
        lens = [len(pair['def_ids']) for pair in batch]
        slen = max(lens)
        def_ids = torch.LongTensor(slen, bs).fill_(2)
        for i in range(bs):
            sent = torch.LongTensor([id for id in batch[i]['def_ids']])
            def_ids[:len(sent), i] = sent
        target_ids = torch.tensor([pair['target_id'] for pair in batch], dtype=torch.int64)
        lens = torch.LongTensor(lens)
        return target_ids, def_ids, lens


def data_processing(data, word2id_xlmr, word2id_target, split_ratio={'train': 0.7, 'valid': 0.1}):
    pdata = []
    for d in data:
        target_id = word2id_target[d['word']]
        sent = ('</s> %s </s>' % d['definitions'].strip()).split()
        tkns = [word2id_xlmr[s if s in word2id_xlmr else UNK_WORD] for s in sent]
        pdata.append({'target_id': target_id, 'def_ids': tkns})
    random.shuffle(pdata)
    return np.split(pdata, [int(len(pdata) * split_ratio['train']), int(len(pdata) * (split_ratio['train'] + split_ratio['valid']))])
