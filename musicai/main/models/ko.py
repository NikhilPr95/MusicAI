from musicai.main.lib.input_vectors import create_standard_feature_matrix
from musicai.main.models.base import Base
from musicai.main.models.omm import OMM
from musicai.main.models.knn import KNN
from musicai.main.constants.values import MAX_NOTES
from musicai.utils.general import *


class KO(Base):
	def __init__(self, data_type, activation=None, kernel=None, ngramlength=None, chords_in_ngram=False, notes=None):
		Base.__init__(self)
		self.knn = KNN()
		self.omm = OMM()
		self.data_type = data_type
		self.activation = activation
		self.kernel = kernel
		self.ngramlength = ngramlength
		self.chords_in_ngram = chords_in_ngram
		self.notes = notes

	def fit(self, bar_sequences, chord_sequences):
		if self.data_type is not None:
			raise Exception("Model does not support {} data type".format(self.data_type))
		if self.activation is not None:
			raise Exception("Model does not support {} activation".format(self.activation))
		if self.kernel is not None:
			raise Exception("Model does not support {} kernel".format(self.kernel))

		bar_sequences_ = flatten(bar_sequences)
		chord_sequences_ = flatten(chord_sequences)
		# print(set([len(x) for x in bar_sequences_]))
		self.knn.fit(bar_sequences_, chord_sequences_)
		self.omm.fit(chord_sequences)

	def predict(self, bar_sequence):
		# print(bar_sequence)
		if len(bar_sequence) < MAX_NOTES:
			bar_sequence += [0] * (MAX_NOTES - len(bar_sequence))  # pad with zeros
		if len(bar_sequence) > MAX_NOTES:
			bar_sequence = bar_sequence[:MAX_NOTES]  # crop perhaps
		# push to shared memory instead of returning here
		knn_result = self.knn.predict(bar_sequence)[0]
		# return knn_result, self.omm.predict(knn_result)
		return self.omm.predict(knn_result)

	def score(self, bar_sequences, chord_sequences):
		bar_notes, knn_labels = create_standard_feature_matrix(bar_sequences, chord_sequences, exclude=1, chord_label_offset=0, num_notes=self.notes)
		_, omm_labels = create_standard_feature_matrix(bar_sequences, chord_sequences, exclude=1, chord_label_offset=1, num_notes=self.notes)
		data, labels = bar_notes, list(zip(knn_labels, omm_labels))

		results = [self.predict(d) for d in data]
		# score1 = sum([1 if r[0] == l[0] else 0 for (r, l) in zip(results, labels)]) / len(data)
		# score2 = sum([1 if r[1] == l[1] else 0 for (r, l) in zip(results, labels)]) / len(data)

		score2 = sum([1 if r == l[1] else 0 for (r, l) in zip(results, labels)]) / len(data)

		# return score1, score2
		return score2