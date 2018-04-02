from musicai.main.models.base import Base
from musicai.main.models.omm import OMM
from musicai.main.models.knn import KNN
from musicai.main.constants.values import MAX_NOTES
from musicai.utils.general import *

class KO(Base):
	def __init__(self):
		Base.__init__(self)
		self.knn = KNN()
		self.omm = OMM()

	def fit(self, bar_sequences, chord_sequences):
		bar_sequences_ = flatten(bar_sequences)
		chord_sequences_ = flatten(chord_sequences)
		print(set([len(x) for x in bar_sequences_]))
		self.knn.fit(bar_sequences_, chord_sequences_)
		self.omm.fit(chord_sequences)

	def predict(self, bar_sequence):
		print(bar_sequence)
		if len(bar_sequence) < MAX_NOTES:
			bar_sequence += [0] * (MAX_NOTES - len(bar_sequence))  # pad with zeros
		if len(bar_sequence) > MAX_NOTES:
			bar_sequence = bar_sequence[:MAX_NOTES]  # crop perhaps
		# push to shared memory instead of returning here
		knn_result = self.knn.predict(bar_sequence)[0]
		return knn_result,self.omm.predict(knn_result)

	def score(self, data, labels):
		results = [self.predict(d) for d in data]
		score1 = sum([1 if r[0] == l[0] else 0 for (r, l) in zip(results, labels)]) / len(data)
		score2 = sum([1 if r[1] == l[1] else 0 for (r, l) in zip(results, labels)]) / len(data)

		return score1, score2