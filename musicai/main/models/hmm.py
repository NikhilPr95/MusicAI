import glob

import numpy as np
from hmmlearn import hmm
from musicai.main.constants import directories
from musicai.main.lib.markov import transition_matrices, emission_matrix
from musicai.main.models.base import Base
from musicai.utils.general import flatten, make_nparray_from_dict

from pickle import load, dump


class HMM(Base):
	def __init__(self):
		Base.__init__(self)
		self.clf = None

	def fit(self, bar_sequences, chord_sequences):
		first_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
		all_chords = flatten(chord_sequences)
		all_notes = flatten(first_notes)
		num_chords = len(set(all_chords))

		model = hmm.MultinomialHMM(num_chords)

		model.startprob, model.transmat = transition_matrices(first_notes)
		emission_dict = emission_matrix(all_notes, all_chords)

		model.emissionprob, notes, chords = make_nparray_from_dict(emission_dict)

		f_note_array = np.concatenate([f for f in first_notes])
		f_note_lengths = [len(f) for f in first_notes]

		f_note_delta = np.array([(f - min(f_note_array)) for f in f_note_array])

		model.fit(f_note_delta.reshape(-1, 1), lengths=f_note_lengths)

		self.clf = model

	def predict(self, notes):

		logprob, val = self.clf.decode(notes)
		return logprob, val
