import json
import torch


class Vocab:
    def __init__(self, vec_path, vocab_words):
        vecs = json.load(open(vec_path, 'r', encoding='utf-8'))
        
        self.id2word = [w for w in vecs.keys() if w in vocab_words]
        self.word2id = {self.id2word[i]: i for i in range(len(self.id2word))}
        self.id2vec = torch.tensor([v for w, v in vecs.items() if w in vocab_words])

        assert len(self.id2word) == len(self.word2id) == len(self.id2vec)
