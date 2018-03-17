import numpy as np
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.main.lib.input_vectors import get_first_note_sequences, ngram_vector, create_ngram_feature_matrix
from musicai.main.models.base import Base
from musicai.utils.general import flatten
from sklearn.neural_network import MLPClassifier


class MLP(Base):
	def __init__(self, ngramlength=4, activation='relu'):
		Base.__init__(self)
		self.clf = None
		self.activation = activation
		# self.activation = 'logistic'
		# self.activation = 'tanh'
		self.activation = 'identity'
		self.ngramlength = ngramlength

	def fit(self, bar_sequences, chord_sequences):
		first_note_sequences = get_first_note_sequences(bar_sequences)
		n = self.ngramlength
		first_note_sequence_ngrams, \
		chord_sequence_ngrams = ngram_vector(first_note_sequences, n), ngram_vector(chord_sequences, n)

		ngram_chord_sequences = flatten(chord_sequence_ngrams)
		ngram_f_note_sequences = flatten(first_note_sequence_ngrams)

		X, y = create_ngram_feature_matrix(ngram_f_note_sequences, ngram_chord_sequences)
		X = np.array(X)
		y = np.array(y)

		self.clf = MLPClassifier(activation=self.activation, max_iter=1000)
		self.clf.fit(X, y)
		print("X shape Y shape", X.shape, y.shape)
		print("score:", self.clf.score(X, y))

	def predict(self, input):
		# chord = self.clf.predict(np.array(input))
		m = np.array(input)
		# print("M", m.shape, m.ndim)
		chord = self.clf.predict([input])
		# print('ch:', chord)
		return SIMPLE_CHORDS[chord[0]]

	# def predict_with_chords(self, notes, chords):
	# 	print('notes:', notes)
	# 	print('chords:', chords)
	#
	# 	chords_numbers = [SIMPLE_CHORDS.index(c) for c in chords]
	# 	chord = self.clf.predict(notes + chords_numbers[:-1])
	# 	return SIMPLE_CHORDS[chord]