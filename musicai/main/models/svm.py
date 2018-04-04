from sklearn import svm

import numpy as np
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.main.lib.input_vectors import create_ngram_feature_matrix, \
	create_classic_feature_matrix
from musicai.main.models.base import Base


class SVM(Base):
	def __init__(self, ngramlength=4, data_type='first_notes', activation=None, kernel='rbf', chords_in_ngram=False):
		Base.__init__(self)
		self.clf = None
		self.ngramlength = ngramlength
		self.data_type = data_type
		self.activation = activation
		self.kernel = kernel
		self.chords_in_ngram = chords_in_ngram

	def fit(self, bar_sequences, chord_sequences):
		if self.activation is not None:
			raise Exception("Model does not support {} activation".format(self.activation))

		if self.data_type == 'first_notes':
			X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, n=self.ngramlength, chords_in_ngram=self.chords_in_ngram)
		elif self.data_type == 'current_bar':
			if self.chords_in_ngram is not False:
				raise Exception("Model does not support chords in ngram with current bar")
			X, y = create_classic_feature_matrix(bar_sequences, chord_sequences)
		else:
			raise Exception("Model does not support {} data type".format(self.data_type))

		X = np.array(X)
		y = np.array(y)

		self.clf = svm.SVC(kernel=self.kernel)
		self.clf.fit(X, y)
		# print("score:", self.clf.score(X, y))

	def predict(self, input):
		chord = self.clf.predict([input])
		return SIMPLE_CHORDS[chord[0]]

	def score(self, bar_sequences, chord_sequences):
		if self.data_type == 'current_bar':
			X, y = create_classic_feature_matrix(bar_sequences, chord_sequences)
		elif self.data_type == 'first_notes':
			X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, n=self.ngramlength, chords_in_ngram=self.chords_in_ngram)
		else:
			raise Exception("Model does not support {} data type".format(self.data_type))

		return self.clf.score(X, y)