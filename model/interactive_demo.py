import pickle
import re
from process_raw_data import filter_query


class Vocab:
	def __init__(self, meta_path):
		self.EOS_token = 3
		with open(meta_path, 'rb') as f:
			meta = pickle.load(f)
			self.vocab = meta['vocab']

	def tokenize(self, sentence):
		tokens = sentence.strip().split()
		tokens = [self.vocab[w] for w in tokens]
		if len(tokens) == 0:
			tokens = [0]
		return tokens

	def pred2words(self, prediction):
		outputs = []
		for pred in prediction:
			new_pred = pred
			for i, x in enumerate(pred):
				if int(x) == self.EOS_token:
					new_pred = pred[:i]
					break
			outputs.append(' '.join([self.vocab[int(x)] for x in new_pred]))
		return outputs


if __name__ == "__main__":
	v = Vocab("./data/reddit_meta.pick")
	text = "hey there , what's up ?"
	filtered_text = filter_query(text)
	print(v.tokenize(filtered_text))
