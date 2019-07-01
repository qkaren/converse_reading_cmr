# -*- coding: utf-8 -*-
# PAge normalization/data extraction logic taken from https://github.com/qkaren/converse_reading_cmr (for consistency)

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString
import re
from nltk.tokenize import TweetTokenizer
from itertools import chain
import pke


class GroudingGenerator:
	def __init__(self, max_fact_len=12, max_facts_count=500, min_fact_len=8):
		self.tokenizer = TweetTokenizer(preserve_case=False)
		self.extractor = pke.unsupervised.TopicRank()
		self.max_fact_len = max_fact_len
		self.max_facts_count = max_facts_count
		self.min_fact_len = min_fact_len

	def insert_escaped_tags(self, tags):
		"""For each tag in "tags", insert contextual tags (e.g., <p> </p>) as escaped text
			so that these tags are still there when html markup is stripped out."""
		found = False
		for tag in tags:
			strs = list(tag.strings)
			if len(strs) > 0:
				l = tag.name
				strs[0].parent.insert(0, NavigableString("<"+l+">"))
				strs[-1].parent.append(NavigableString("</"+l+">"))
				found = True
		return found

	def norm_fact(self, t, tokenize=True):
		# Minimalistic processing: remove extra space characters
		t = re.sub("[ \n\r\t]+", " ", t)
		t = t.strip()
		if tokenize:
			t = " ".join(self.tokenizer.tokenize(t))
			t = t.replace('[ deleted ]','[deleted]');
		# Preprocessing specific to fact
		t = self.filter_text(t)
		t = re.sub('- wikipedia ', '', t, 1)
		t = re.sub(' \[ edit \]', '', t, 1)
		t = re.sub('<h2> navigation menu </h2>', '', t)
		return t

	def norm_article(self, t):
		"""Minimalistic processing with linebreaking."""
		t = re.sub("\s*\n+\s*","\n", t)
		t = re.sub(r'(</[pP]>)',r'\1\n', t)
		t = re.sub("[ \t]+"," ", t)
		t = t.strip()
		return t

	def get_wiki_page_url(self, title):
		"""Search for wiki URL for given topic"""
		r = requests.get("https://en.wikipedia.org/w/api.php?action=opensearch&search=%s&limit=1&format=json" % "%20".join(title.split(" "))).json()[3]
		if len(r) == 0:
			return None
		wanted_url = r[0]
		main_content = requests.get(wanted_url)
		return main_content

	def get_desired_content(self, page_content):
		"""Return facts extracted from website"""
		notext_tags = ['script', 'style']
		important_tags = ['title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'p']
		b = BeautifulSoup(page_content,'html.parser')
		# Remove tags whose text we don't care about (javascript, etc.):
		for el in b(notext_tags):
			el.decompose()
		# Delete other unimportant tags, but keep the text:
		for tag in b.findAll(True):
			if tag.name not in important_tags:
				tag.append(' ')
				tag.replaceWithChildren()
		# All tags left are important (e.g., <p>) so add them to the text: 
		self.insert_escaped_tags(b.find_all(True))
		# Extract facts from html:
		t = b.get_text(" ")
		t = self.norm_article(t)
		facts = []
		for sent in filter(None, t.split("\n")):
			if len(sent.split(" ")) >= self.min_fact_len:
				facts.append(self.process_fact(sent))
		return self.combine_facts(facts)

	def filter_text(self, text):
		#https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
		text = re.sub(r'[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?', '<NUM>', text)
		text = re.sub("[\(].*?[\)]", "", text)    # [\[\(].*?[\]\)]
		text = text.split()
		new_text = []
		for x in text:
			if 'www' in x or 'http' in x:
				continue
			new_text.append(x)
		return ' '.join(new_text)

	def process_fact(self, fact):
		fact = self.filter_text(self.norm_fact(fact)).strip().split()
		if len(fact) > 100:
			fact = fact[:100] + ['<TRNC>']
		return fact

	def combine_facts(self, facts):
		facts = facts[:self.max_fact_len]
		facts = ' '.join(list(chain(*facts)))
		facts = facts.split()
		if len(facts) == 0:
			facts = ['UNK']
		if len(facts) > self.max_facts_count:
			facts = facts[:self.max_facts_count] + ['<TRNC>']
		return facts

	def topic_extraction(self, text):
		self.extractor.load_document(input=text, language='en')
		self.extractor.candidate_selection()
		self.extractor.candidate_weighting()
		keyphrases = self.extractor.get_n_best(n=10)
		return keyphrases

	def get_appropriate_grounding(self, topics):
		for topic in topics:
			x = self.get_wiki_page_url(topic[0])
			if x:
				return x
		return None

	def get_grounding_data(self, text):
		topics = self.topic_extraction(text)
		url = self.get_appropriate_grounding(topics)
		grounding = self.get_desired_content(url.content)
		return grounding


if __name__ == "__main__":
	print("Generating grounding data. This may take a while...")
	conversation = "hey thhere, what is up? I love Nokia phones."
	g = GroudingGenerator()
	grounding = g.get_grounding_data(conversation)
	print(grounding)
