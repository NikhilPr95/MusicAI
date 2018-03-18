import random
from nltk.util import ngrams
from nltk.probability import FreqDist
from musicai.utils.general import *

def get_unigram(gram_dict):
	return random.choice([key for key in gram_dict.keys()])


def get_bigram_from_prefix(gram_dict, prefix):
	return random.choice([key for key in gram_dict.keys() if key[0][0] == prefix[1][0]])


def get_trigram_from_prefix(gram_dict, prefix):
	print(prefix)
	print(gram_dict)
	print([i[0] for i in prefix[-2:]])
	return random.choice([key for key in gram_dict.keys() if [i[0] for i in key[0:2]] == [i[0] for i in prefix[-2:]]])


class Text:
	def __init__(self, doc):
		self.tokens = doc
		self.vocabulary = list(range(60,72))
		self.unigrams = list(ngrams(self.tokens, 1))
		self.bigrams = list(ngrams(self.tokens, 2))
		self.trigrams = list(ngrams(self.tokens, 3))
		self.unigram_types = len(set(self.unigrams))
		self.bigram_types = len(set(self.bigrams))
		self.trigram_types = len(set(self.trigrams))
		self.unigram_tokens = len(self.unigrams)
		self.bigram_tokens = len(self.bigrams)
		self.trigram_tokens = len(self.trigrams)
		self.unigram_counts = FreqDist(self.unigrams)
		self.bigram_counts = FreqDist(self.bigrams)
		self.trigram_counts = FreqDist(self.trigrams)
		self.unigram_probabilities = {}
		self.bigram_probabilities = {}
		self.trigram_probabilities = {}
		self.unigram_notes = (list(ngrams([i[0] for i in self.tokens], 1)))
		self.bigram_notes = (list(ngrams([i[0] for i in self.tokens], 2)))
		self.trigram_notes = (list(ngrams([i[0] for i in self.tokens], 3)))
		self.unigram_notes_counts = FreqDist(self.unigram_notes)
		self.bigram_notes_counts = FreqDist(self.bigram_notes)
		self.trigram_notes_counts = FreqDist(self.trigram_notes)
		self.compute_probabilites()
		#self.add_one_smoothing()
	
	def compute_probabilites(self):
		for i in self.unigrams:
			self.unigram_probabilities[i] = (self.unigram_notes_counts[(i[0][0],)]+1) / (len(self.unigram_notes) +  len(self.vocabulary))
		for i in self.bigrams:
			self.bigram_probabilities[i] = (self.bigram_notes_counts[(i[0][0],i[1][0])]+1) / (self.unigram_notes_counts[(i[0][0],)] + len(self.vocabulary) - 1)
		for i in self.trigrams:
			self.trigram_probabilities[i] = (self.trigram_notes_counts[(i[0][0],i[1][0],i[2][0])]+1) / (self.bigram_notes_counts[(i[0][0], i[1][0])] + len(self.vocabulary) -2)
		
		print(self.bigram_probabilities)
	
	def set_counts(self, ngrams):
		counts = dict()
		print(len(set(ngrams)))
		for ngram in set(ngrams):
			counts[ngram] = ngrams.count(ngram)
		return counts
	
	
	def generate_unigram_sequences(self, n):
		generated_seq = []
		
		while (len(generated_seq) < n):
			generated_seq.append(get_unigram(self.unigram_counts))
		return generated_seq
	
	def generate_bigram_sequences(self, n):
		generated_seq = []
		generated_seq.append(get_bigram_from_prefix(self.bigram_probabilities, random.choice(list(self.bigram_probabilities.keys()))))
		while (len(generated_seq) < n):
			try:
				generated_seq.append(get_bigram_from_prefix(self.bigram_probabilities, generated_seq[-1]))
			except:
				generated_seq.append(get_bigram_from_prefix(self.bigram_probabilities, random.choice(list(self.bigram_probabilities.keys()))))
		
		return generated_seq
	
	def generate_trigram_sequences(self, n):
		generated_seq = []
		generated_seq.append(get_trigram_from_prefix(self.trigram_probabilities, random.choice(list(self.trigram_probabilities.keys()))))
		while (len(generated_seq) < n):
			try:
				generated_seq.append(get_trigram_from_prefix(self.trigram_probabilities, generated_seq[-1]))
			except:
				generated_seq.append(get_bigram_from_prefix(self.trigram_probabilities, random.choice(list(self.trigram_probabilities.keys()))))
		
		return generated_seq[1:]
	
	
	def add_one_smoothing(self):
		for gram in list(ngrams(self.vocabulary,1)):
			if gram not in self.unigram_probabilities:
				self.unigram_probabilities[gram] = 1 / (len(self.unigrams) + len(self.vocabulary))
		
		for gram in list(ngrams(self.vocabulary,2)):
			if gram not in self.bigram_probabilities:
				self.bigram_probabilities[gram] = 1 / (len(self.bigrams) + len(self.vocabulary) - 1)
				
		for gram in list(ngrams(self.vocabulary,3)):
			if gram not in self.trigram_probabilities:
				self.trigram_probabilities[gram] = 1 / (len(self.trigrams) + len(self.vocabulary) - 2)