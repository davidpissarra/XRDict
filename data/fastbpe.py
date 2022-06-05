import fastBPE
import json
import random

data = json.load(open('train_full.json', 'r' ,encoding='utf-8'))
random.shuffle(data)

bpe = fastBPE.fastBPE('codes_xnli_15', 'vocab_xnli_15')

for d in data:
    d['definitions'] = bpe.apply([d['definitions']])[0]

json.dump(data, open("bpe_full_15.json", "w", encoding='utf-8'), indent=4, ensure_ascii=False)
