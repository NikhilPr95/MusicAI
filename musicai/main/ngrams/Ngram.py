import random
from nltk.util import ngrams
from nltk.probability import FreqDist


def get_unigram(gram_dict):
	return random.choice([key for key in gram_dict.keys()])


def get_bigram_from_prefix(gram_dict, prefix):
	print(gram_dict, prefix)
	return random.choice([key for key in gram_dict.keys() if key[0] == prefix[1]])


def get_bigrams_with_prefix(gram_dict, prefix):
	print(gram_dict,prefix)
	return ([key for key in gram_dict.keys() if key[0] == prefix[1]])


def get_trigram_from_prefix(gram_dict, prefix):
	return random.choice([key for key in gram_dict.keys() if key[0:2] == prefix[-2:]])


def get_trigrams_with_prefix(gram_dict, prefix):
	return ([key for key in gram_dict.keys() if key[0:2] == prefix[-2:]])

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
		
		self.compute_probabilites()
		self.add_one_smoothing()
	
	def compute_probabilites(self):
		for i in self.unigrams:
			self.unigram_probabilities[i] = self.unigram_counts[i]+1 / (len(self.unigrams) +  len(self.vocabulary))
		for i in self.bigrams:
			self.bigram_probabilities[i] = self.bigram_counts[i]+1 / (self.unigram_counts[(i[0],)] + len(self.vocabulary) - 1)
		for i in self.trigrams:
			self.trigram_probabilities[i] = self.trigram_counts[i]+1/ (self.bigram_counts[(i[0], i[1])] + len(self.vocabulary) -2)
		
		print(self.unigram_probabilities)
	
	def set_counts(self, ngrams):
		counts = dict()
		print(len(set(ngrams)))
		for ngram in set(ngrams):
			counts[ngram] = ngrams.count(ngram)
		return counts
	
	def get_unigram_prob(self, unigram):
		return self.unigram_counts[unigram] / len(self.unigrams)
	
	def get_bigram_prob(self, bigram):
		return self.bigram_counts[bigram] / self.unigram_counts[(bigram[0],)]
	
	def get_trigram_prob(self, trigram):
		return self.trigram_counts[trigram] / self.bigram_counts[(trigram[0], trigram[1])]
	
	def generate_unigram_sequences(self, n):
		generated_seq = []
		
		while (len(generated_seq) < n):
			generated_seq.append(get_unigram(self.unigram_counts))
		return generated_seq
	
	def generate_bigram_sequences(self, n):
		generated_seq = []
		generated_seq.append(get_bigram_from_prefix(self.bigram_counts, random.choice(list(self.bigram_probabilities.keys()))))
		while (len(generated_seq) < n):
			try:
				generated_seq.append(get_bigram_from_prefix(self.bigram_counts, generated_seq[-1]))
			except:
				generated_seq.append(get_bigram_from_prefix(self.bigram_counts, random.choice(list(self.bigram_counts.keys()))))
		
		return generated_seq
	
	def generate_trigram_sequences(self, n):
		generated_seq = []
		generated_seq.append(get_bigram_from_prefix(self.bigram_counts, random.choice(list(self.bigram_probabilities.keys()))))
		while (len(generated_seq) < n):
			try:
				generated_seq.append(get_trigram_from_prefix(self.trigram_counts, generated_seq[-1]))
			except:
				generated_seq.append(get_bigram_from_prefix(self.bigram_counts, random.choice(list(self.trigram_counts.keys()))))
		
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