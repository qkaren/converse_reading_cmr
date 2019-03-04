import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .recurrent import OneLayerBRNN, ContextualEmbed
from .dropout_wrapper import DropoutWrapper
from .encoder import LexiconEncoder
from .similarity import DeepAttentionWrapper, FlatSimilarityWrapper, SelfAttnWrapper
from .similarity import AttentionWrapper
from .san_decoder import SANDecoder

class DNetwork_Seq2seq(nn.Module):
    """Network for Seq2seq/Memnet doc reader."""
    def __init__(self, opt, embedding=None, padding_idx=0):
        super(DNetwork_Seq2seq, self).__init__()
        my_dropout = DropoutWrapper(opt['dropout_p'], opt['vb_dropout'])
        self.dropout = my_dropout

        self.lexicon_encoder = LexiconEncoder(opt, embedding=embedding, dropout=my_dropout)
        query_input_size = self.lexicon_encoder.query_input_size
        doc_input_size = self.lexicon_encoder.doc_input_size

        print('Lexicon encoding size for query and doc are:{}', doc_input_size, query_input_size)
        covec_size = self.lexicon_encoder.covec_size
        embedding_size = self.lexicon_encoder.embedding_dim
        # share net
        contextual_share = opt.get('contextual_encoder_share', False)
        prefix = 'contextual'
        prefix = 'contextual'
        # doc_hidden_size
        self.doc_encoder_low = OneLayerBRNN(doc_input_size + covec_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
        self.doc_encoder_high = OneLayerBRNN(self.doc_encoder_low.output_size + covec_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
        if contextual_share:
            self.query_encoder_low = self.doc_encoder_low
            self.query_encoder_high = self.doc_encoder_high
        else:
            self.query_encoder_low = OneLayerBRNN(query_input_size + covec_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)
            self.query_encoder_high = OneLayerBRNN(self.query_encoder_low.output_size + covec_size, opt['contextual_hidden_size'], prefix=prefix, opt=opt, dropout=my_dropout)

        doc_hidden_size = self.doc_encoder_low.output_size + self.doc_encoder_high.output_size
        query_hidden_size = self.query_encoder_low.output_size + self.query_encoder_high.output_size

        self.query_understand = OneLayerBRNN(query_hidden_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        doc_attn_size = doc_hidden_size + covec_size + embedding_size
        query_attn_size = query_hidden_size + covec_size + embedding_size
        num_layers = 3

        prefix = 'deep_att'
        self.deep_attn = DeepAttentionWrapper(doc_attn_size, query_attn_size, num_layers, prefix, opt, my_dropout)

        doc_und_size = doc_hidden_size + query_hidden_size + self.query_understand.output_size
        self.doc_understand = OneLayerBRNN(doc_und_size, opt['msum_hidden_size'], prefix='msum', opt=opt, dropout=my_dropout)
        query_mem_hidden_size = self.query_understand.output_size
        doc_mem_hidden_size = self.doc_understand.output_size

        if opt['self_attention_on']:
            att_size = embedding_size + covec_size + doc_hidden_size + query_hidden_size + self.query_understand.output_size + self.doc_understand.output_size
            self.doc_self_attn = AttentionWrapper(att_size, att_size, prefix='self_att', opt=opt, dropout=my_dropout)
            doc_mem_hidden_size = doc_mem_hidden_size * 2
            self.doc_mem_gen = OneLayerBRNN(doc_mem_hidden_size, opt['msum_hidden_size'], 'msum', opt, my_dropout)
            doc_mem_hidden_size = self.doc_mem_gen.output_size
        # Question merging
        self.query_sum_attn = SelfAttnWrapper(query_mem_hidden_size, prefix='query_sum', opt=opt, dropout=my_dropout)
        self.decoder = SANDecoder(doc_mem_hidden_size, query_mem_hidden_size, opt, prefix='decoder', dropout=my_dropout)
        self.opt = opt
        self.hidden_size = self.query_understand.output_size
        self.gru = nn.GRUCell(embedding_size, self.hidden_size)
        self.embedding = nn.Embedding(opt['vocab_size'], embedding_size, padding_idx=0)
        self.memA = nn.Linear(embedding_size, self.hidden_size, bias=False)
        self.memC = nn.Linear(embedding_size, self.hidden_size, bias=False)

    def forward(self, input, hidden):
        hidden = self.gru(input, hidden)
        return hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def _get_doc_sentence_embeddings(self, batch):
        sentence_end_tok_ids = [' 5 ', ' 157 ', ' 80 ', ' 180 ']

        batch_size = len(batch['doc_tok'])

        doc_sents = []
        max_num_sents = -1
        max_sent_len = -1
        max_doc_len = -1

        doc = Variable(torch.cat([batch['doc_tok'], torch.LongTensor([[2, 5, 1, 2]] * batch_size)], dim=1)).data.numpy()
        for doc_i in doc: # i-th example, e.g., [9 9 5 8 8]

            max_doc_len = max(max_doc_len, len(doc_i))

            doc_i_str = ' '.join([str(_) for _ in doc_i]) # '9 9 5 8 8'
            for se in sentence_end_tok_ids:
                doc_i_str = doc_i_str.replace(se, '<SPLIT>') # ['9 9 <SPLIT> 8 8']
            doc_i_str_split = doc_i_str.split('<SPLIT>') # ['9 9', '8 8']
            doc_i_str_toks = [_.strip().split() for _ in doc_i_str_split] # [['9', '9'], ['8', '8']]

            num_sent = len(doc_i_str_toks)
            max_num_sents = max(num_sent, max_num_sents)

            max_sent_len_i = max(len(_) for _ in doc_i_str_toks)
            max_sent_len = max(max_sent_len, max_sent_len_i)

            def _doc_ij_str_to_idx(doc_ij_str):
                return [int(_) for _ in doc_ij_str]

            doc_sents.append([_doc_ij_str_to_idx(_) for _ in doc_i_str_toks])
                # [..., [[9, 9], [8, 8]], ...]

        doc_sents_tensor = torch.LongTensor(batch_size, max_num_sents, max_sent_len).fill_(0)
        for i, doc_i in enumerate(doc_sents):
            for j, doc_ij in enumerate(doc_i):
                sent_len_ij = len(doc_ij)
                doc_sents_tensor[i, j, :sent_len_ij] = torch.LongTensor(doc_ij)

        if self.opt['cuda']:
            doc_sents_tensor = doc_sents_tensor.cuda()
        doc_sents_tensor = Variable(doc_sents_tensor)
        doc_sents_emb = self.embedding(doc_sents_tensor.view(batch_size, -1))
        doc_sents_emb = doc_sents_emb.view(batch_size, max_num_sents, max_sent_len, -1)
        doc_sents_emb = torch.mean(doc_sents_emb, dim=2)
        return doc_sents_emb

    def add_fact_memory(self, query_final_hidden, batch):
        doc_emb = self._get_doc_sentence_embeddings(batch)

        m = self.memA(doc_emb)
        c = self.memC(doc_emb)

        u = query_final_hidden.unsqueeze(1)
        attn = m * u
        attn = torch.sum(attn, dim=-1).squeeze()
        attn = F.softmax(attn)
        mem_hidden = torch.sum(attn.unsqueeze(2) * c, dim=1).squeeze()
        return mem_hidden
