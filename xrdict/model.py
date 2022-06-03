import torch

from xlm.utils import AttrDict
from xlm.model.transformer import TransformerModel
from xlm.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD


class XRDict(torch.nn.Module):
    def __init__(self, ckpt_path, vocab):
        super().__init__()
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.embedding = torch.nn.Embedding.from_pretrained(vocab.id2vec)
        self.embedding.weight.requires_grad = False

        reloaded = torch.load(ckpt_path)
        self.params = AttrDict(reloaded['params'])

        self.dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        self.params.n_words = len(self.dico)
        self.params.bos_index = self.dico.index(BOS_WORD)
        self.params.eos_index = self.dico.index(EOS_WORD)
        self.params.pad_index = self.dico.index(PAD_WORD)
        self.params.unk_index = self.dico.index(UNK_WORD)
        self.params.mask_index = self.dico.index(MASK_WORD)

        self.encoder = TransformerModel(self.params, self.dico, True, True)
        self.encoder.load_state_dict(reloaded['model'])

        self.fc = torch.nn.Linear(self.encoder.dim, self.embedding.embedding_dim)

    
    def forward(self, x, lengths, causal=False, word_gt=None, mode='test'):
        # sentence vectors: Tensor(batch_size, encoder_dim)      
        h0 = self.encoder('fwd', x=x, lengths=lengths, causal=causal)[0]

        # Tensor(batch_size, embedding_dim)
        h0 = self.fc(h0)

        # score: Tensor(batch, vocab_size)
        score = torch.mm(h0, self.embedding.weight.T)

        # sorted score index: Tensor(batch, vocab_size)
        _, word_ids = torch.sort(score, dim=1, descending=True)

        if mode == 'train':
            loss = self.loss(score, word_gt)
            return loss, score, word_ids
        elif mode == 'test':
            return word_ids
