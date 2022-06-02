import json
import torch


class Vocab:
    def __init__(self, path):
        f = open(path, 'r', encoding='utf-8')
        vecs = json.load(f)
        f.close()

        self.id2word = list(vecs.keys())
        self.word2id = {self.id2word[i]: i for i in range(len(self.id2word))}
        self.id2vec = torch.tensor([vec[1] for vec in vecs.items()])
