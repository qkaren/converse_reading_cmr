import pickle
import os
import re
import torch as th
import numpy as np

from process_raw_data import filter_query, filter_fact
from src.batcher import load_meta, prepare_batch_data
from src.model import DocReaderModel
from config import set_args
from src.fetch_realtime_grounding import GroudingGenerator


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


class InteractiveModel:
	def __init__(self, args):
		self.is_cuda = args.cuda
		self.embedding, self.opt, self.vocab = load_meta(vars(args), args.meta)
		self.opt['skip_tokens'] = self.get_skip_tokens(self.opt["skip_tokens_file"])
		self.opt['skip_tokens_first'] = self.get_skip_tokens(self.opt["skip_tokens_first_file"])
		self.state_dict = th.load(args.model_dir)["state_dict"]
		self.model = DocReaderModel(self.opt, self.embedding, self.state_dict)
		self.model.setup_eval_embed(self.embedding)
		if self.is_cuda:
			self.model.cuda()

	def get_skip_tokens(self, path):
		skip_tokens = None
		if path and os.path.isfile(path):
			skip_tokens = []
			with open(path, 'r') as f:
				for word in f:
					word = word.strip().rstrip('\n')
					try:
						skip_tokens.append(self.vocab[word])
					except:
						print("Token %s not present in dictionary" % word)
		return skip_tokens

	def predict(self, data, top_k=2):
		processed_data = prepare_batch_data([self.preprocess_data(x) for x in data], ground_truth=False)
		prediction, prediction_topks = self.model.predict(processed_data, top_k=top_k)
		pred_word = pred2words(prediction, self.vocab)
		prediction = [np.asarray(x, dtype=np.str).tolist() for x in pred_word]
		return (prediction, prediction_topks)

	def preprocess_data(self, sample, q_cutoff=30, doc_cutoff=500):
		def tok_func(toks):
			return [self.vocab[w] for w in toks]

		fea_dict = {}

		query_tokend = filter_query(sample['query'].strip(), max_len=q_cutoff).split()
		doc_tokend = filter_fact(sample['fact'].strip()).split()
		if len(doc_tokend) > doc_cutoff:
			doc_tokend = doc_tokend[:doc_cutoff] + ['<TRNC>']

		# TODO
		fea_dict['query_tok'] = tok_func(query_tokend)
		fea_dict['query_pos'] = []
		fea_dict['query_ner'] = []

		fea_dict['doc_tok'] = tok_func(doc_tokend)
		fea_dict['doc_pos'] = []
		fea_dict['doc_ner'] = []
		fea_dict['doc_fea'] = ''

		if len(fea_dict['query_tok']) == 0:
			fea_dict['query_tok'] = [0]
		if len(fea_dict['doc_tok']) == 0:
			fea_dict['doc_tok'] = [0]

		return fea_dict


if __name__ == "__main__":
	import time
	args = set_args()
	t = time.time()
	m = InteractiveModel(args)
	t = time.time() - t
	print("[LOG] Time taken to load model: %.3fs" % t)
	grounding = input(">> Enter grounding: ")
	conversations = []
	for _ in range(3):
		conversation = input(">> Enter query: ")
		conversations.append(conversation)
		context = "START EOS " + " EOS ".join(conversations)
		# Generate predictions
		data = [{'query': context, 'fact': grounding}]
		t = time.time()
		prediction = m.predict(data, top_k=args.decoding_topk)[0][0]
		t = time.time() - t
		conversations.append(prediction)
		print("[LOG] Time taken to generate predictions: %.3fs" % t)
		print(">> Response: %s " % prediction)
