from sklearn import svm

import numpy as np
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.main.lib.input_vectors import create_ngram_feature_matrix, \
	create_classic_feature_matrix
from musicai.main.models.base import Base


class SVM(Base):
	def __init__(self, ngramlength=5, data_type='first_notes'):
		Base.__init__(self)
		self.clf = None
		# self.activation = 'logistic'
		# self.activation = 'tanh'
		# self.activation = 'identity'
		self.ngramlength = ngramlength
		self.data_type = data_type

	def fit(self, bar_sequences, chord_sequences):
		X, y = [], []
		if self.data_type == 'first_notes':
			X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, n=self.ngramlength)
		elif self.data_type == 'current_bar':
			X, y = create_classic_feature_matrix(bar_sequences, chord_sequences)

		X = np.array(X)
		y = np.array(y)

		self.clf = svm.SVC()
		self.clf.fit(X, y)
		print("X shape Y shape", X.shape, y.shape)
		print("score:", self.clf.score(X, y))

	def predict(self, input):
		m = np.array(input)
		chord = self.clf.predict([input])
		return SIMPLE_CHORDS[chord[0]]

	def score(self, X, y):
		return self.clf.score(X, y)