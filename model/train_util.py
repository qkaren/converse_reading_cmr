import json
import numpy as np
import subprocess
from my_utils.squad_eval import get_bleu_moses
import os.path


def pred2words(prediction, vocab):
    EOS_token = 3
    outputs = []
    for pred in prediction:
        new_pred = pred
        for i, x in enumerate(pred):
            if int(x) == EOS_token:
                new_pred = pred[:i]
                break
        outputs.append(' '.join([vocab[int(x)] for x in new_pred]))
    return outputs


def get_answer(path):
    with open(path, encoding="utf8") as f:
        answers = json.load(f)
    return answers

def eval_test_loss(model, dev_batches):
    dev_batches.reset()

    tot_loss = 0
    num_data = 0
    for batch in dev_batches:
        loss = model.eval_test_loss(batch)
        batch_size = len(batch['answer_token'])
        num_data += batch_size
        tot_loss += loss * batch_size

    return tot_loss / num_data

def compute_diversity(hypotheses,output_path):
    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", './bleu_eval/diversity.pl.remove_extension', output_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe.encode())
    pipe.stdin.close()
    return pipe.stdout.read()

def check(model, dev_batches, vocab, full_path='',
          output_path='', full_output=''):
    dev_batches.reset()
    predictions = []
    pred_words = []
    dev_toks_tmp = []
    dev_fact_tmp = []
    dev_toks = []
    dev_fact = []

    for batch in dev_batches:
        prediction, prediction_topks = model.predict(batch)
        pred_word = pred2words(prediction, vocab)
        prediction = [np.asarray(x, dtype=np.str).tolist() for x in prediction]
        predictions += prediction
        pred_words += pred_word
        dev_toks_tmp += batch['answer_token'].numpy().tolist()
        dev_fact_tmp += batch['doc_tok'].numpy().tolist()


    for t, f in zip(dev_toks_tmp, dev_fact_tmp):
        t = t[1:t.index(3)]
        if 0 in f:
            f = f[:f.index(0)]
        dev_toks.append(t)
        dev_fact.append(f)
    dev_toks = pred2words(dev_toks, vocab)
    dev_fact = pred2words(dev_fact, vocab)
    dev_toks_list = [line.strip().split(' ') for line in dev_toks]
    pred_words_list = [line.strip().split(' ') for line in pred_words]
    dev_fact_list = [line.strip().split(' ') for line in dev_fact]
    bleu_result = get_bleu_moses(pred_words_list, dev_toks_list)
    bleu_fact = get_bleu_moses(dev_fact_list, pred_words_list)
    bleu = str(bleu_result).split('=')
    bleu = bleu[1].split(',')[0]
    bleu_fact = str(bleu_fact).split('=')
    bleu_fact = bleu_fact[1].split(',')[0]

    with open(output_path, 'w') as f:
        for hypothesis in pred_words_list:
            f.write(' '.join(hypothesis) + '\n')

    with open(full_path, 'r', encoding='utf8') as fr:
        full_lines = fr.readlines()
        assert (len(full_lines) == len(pred_words_list))
        with open(full_output, 'w', encoding='utf8') as fw:
            for f, g in zip(full_lines,pred_words_list):
                f = f.strip().split('\t')
                f[-1] = ' '.join(g).strip()
                f = '\t'.join(f)
                fw.write(f + '\n')
                fw.flush()

    diversity = compute_diversity(pred_words_list, output_path)
    diversity = str(diversity).strip().split()
    diver_uni = diversity[0][2:]
    diver_bi = diversity[1][:-3]
    return bleu, bleu_fact, diver_uni, diver_bi

def write_test_metrics(model_name, dstc_dict, path_report):
    d = dstc_dict
    n_lines = d['n_lines']
    nist = d['nist']
    bleu = d['bleu']
    meteor = d['meteor']
    entropy = d['entropy']
    div = d['diversity']
    avg_len = d['avg_len']

    results = [n_lines] + nist + bleu + [meteor] + entropy + div + [avg_len]

    if not os.path.isfile(path_report):
        with open(path_report, 'w') as f:
            f.write('\t'.join(
                ['fname', 'n_lines'] + \
                ['nist%i' % i for i in range(1, 4 + 1)] + \
                ['bleu%i' % i for i in range(1, 4 + 1)] + \
                ['meteor'] + \
                ['entropy%i' % i for i in range(1, 4 + 1)] + \
                ['div1', 'div2', 'avg_len']
            ) + '\n')

    with open(path_report, 'a') as f:
        f.write('\t'.join(map(str, [model_name] + results)) + '\n')
