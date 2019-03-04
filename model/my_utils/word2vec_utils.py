import numpy as np
from .tokenizer import normalize_text

def load_glove_vocab(path, glove_dim, wv_dim=300):
    vocab = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def build_embedding(path, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, wv_dim))
    emb[0] = 0
    count = 0

    w2id = {w: i for i, w in enumerate(targ_vocab)}

    with open(path, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
                count += 1
    print ('loading glove done!')
    print (count)
    return emb
