from xlm.data.dictionary import UNK_WORD
import torch

device = torch.device('cuda')


class XRDictDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, i):
        return self.pairs[i]


def data_processing(data, word2id_xlmr, word2id_target, split_ratio={'train': 0.7, 'valid': 0.1}):
    pdata = list()
    for d in data:
        if d['word'] in word2id_target:
            target_id = word2id_target[d['word']]
        else:
            continue
        
        tkns = list()
        sent = ('</s> %s </s>' % d['definitions'].strip()).split()
        for s in sent:
            if s in word2id_xlmr:
                tkns.append(word2id_xlmr[s])
            else:
                tkns.append(word2id_xlmr[UNK_WORD])

        pdata.append({
            'target_id': target_id,
            'def_ids': tkns
        })

    n_train = int(len(pdata) * split_ratio['train'])
    n_valid = int(len(pdata) * (split_ratio['train'] + split_ratio['valid']))

    return pdata[:n_train], pdata[n_train:n_valid], pdata[n_valid:]

def xrdict_collate_fn(batch):
    bs = len(batch)
    lens = [len(pair['def_ids']) for pair in batch]
    slen = max(lens)

    def_ids = torch.LongTensor(slen, bs).fill_(2).to(device)
    for i in range(bs):
        sent = torch.LongTensor([id for id in batch[i]['def_ids']])
        def_ids[:len(sent), i] = sent

    target_ids = torch.tensor([pair['target_id'] for pair in batch], dtype=torch.int64, device=device)
    lens = torch.LongTensor(lens).to(device)

    return target_ids, def_ids, lens
