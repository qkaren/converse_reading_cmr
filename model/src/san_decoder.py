import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils import weight_norm
from torch.nn import AlphaDropout
import numpy as np
from functools import wraps
from src.common import activation
from src.similarity import FlatSimilarityWrapper
from src.recurrent import RNN_MAP
from src.dropout_wrapper import DropoutWrapper

SMALL_POS_NUM=1.0e-30
RNN_MAP = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

def generate_mask(new_data, dropout_p=0.0):
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1)-1)
        new_data[i][one] = 1
    mask = Variable(1.0/(1 - dropout_p) * torch.bernoulli(new_data), requires_grad=False)
    return mask

class SANDecoder(nn.Module):
    def __init__(self, x_size, h_size, opt={}, prefix='answer', dropout=None):
        super(SANDecoder, self).__init__()
        self.prefix = prefix
        self.attn_b  = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.attn_e  = FlatSimilarityWrapper(x_size, h_size, prefix, opt, dropout)
        self.rnn_type = opt.get('{}_rnn_type'.format(prefix), 'gru')
        self.rnn =RNN_MAP.get(self.rnn_type, nn.GRUCell)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.answer_opt = opt.get('{}_opt'.format(prefix), 0)
        # 0: std mem; 1: random select step; 2 random selection; voting in pred; 3:sort merge
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.gamma = opt.get('{}_mem_gamma'.format(prefix), 0.5)
        self.alpha = Parameter(torch.zeros(1, 1, 1))

        self.proj = nn.Linear(h_size, x_size) if h_size != x_size else None
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.h2h = nn.Linear(h_size, h_size)
        self.a2h = nn.Linear(x_size, h_size, bias=False)
        self.luong_output_layer = nn.Linear(h_size + x_size, h_size)

    def forward(self, input, hidden, context, context_mask):
        #print(input.size(), hidden.size(), context.size(), context_mask.size())
        hidden = self.dropout(hidden)
        hidden = self.rnn(input, hidden)

        if self.opt['model_type'] == 'san':
            attn = self.attention(context, hidden, context_mask)
            attn_h = torch.cat([hidden, attn], dim=1)
            new_hidden = F.tanh(self.luong_output_layer(attn_h))
        elif self.opt['model_type'] in {'seq2seq', 'memnet'}:
            new_hidden = hidden
        else:
            raise ValueError('Unknown model type: {}'.format(self.opt['model_type']))

        return new_hidden

    def attention(self, x, h0, x_mask):
        if self.answer_opt in {1, 2, 3}:
            st_scores = self.attn_b(x, h0, x_mask)
            if self.answer_opt == 3:
                ptr_net_b = torch.bmm(F.softmax(st_scores).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                xb = ptr_net_b if self.proj is None else self.proj(ptr_net_b)
                end_scores = self.attn_e(x, h0 + xb, x_mask)
                ptr_net_e = torch.bmm(F.softmax(end_scores).unsqueeze(1), x).squeeze(1)
                ptr_net_in = (ptr_net_b + ptr_net_e)/2.0
            elif self.answer_opt == 2:
                ptr_net_b = torch.bmm(F.softmax(st_scores).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                xb = ptr_net_b if self.proj is None else self.proj(ptr_net_b)
                end_scores = self.attn_e(x, xb, x_mask)
                ptr_net_e = torch.bmm(F.softmax(end_scores).unsqueeze(1), x).squeeze(1)
                ptr_net_in = ptr_net_e
            elif self.answer_opt == 1:
                ptr_net_b = torch.bmm(F.softmax(st_scores).unsqueeze(1), x).squeeze(1)
                ptr_net_b = self.dropout(ptr_net_b)
                ptr_net_in = ptr_net_b
        else:
            end_scores = self.attn_e(x, h0, x_mask)
            ptr_net_e = torch.bmm(F.softmax(end_scores).unsqueeze(1), x).squeeze(1)
            ptr_net_in = ptr_net_e

        return ptr_net_in
