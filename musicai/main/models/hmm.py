import glob

import numpy as np
from hmmlearn import hmm
from musicai.main.constants import directories
from musicai.main.constants.values import CHORDS
from musicai.main.lib.markov import transition_matrices, emission_matrix
from musicai.main.models.base import Base
from musicai.utils.general import flatten, make_nparray_from_dict

from pickle import load, dump


class HMM(Base):
	def __init__(self):
		Base.__init__(self)
		self.clf = None
		self.chords = None

	def fit(self, bar_sequences, chord_sequences):
		first_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
		all_chords = flatten(chord_sequences)
		all_notes = flatten(first_notes)
		num_chords = len(set(all_chords))

		model = hmm.MultinomialHMM(num_chords)

		startprobs, transmat = transition_matrices(chord_sequences)
		emission_dict = emission_matrix(all_chords, all_notes)

		for key in CHORDS:
			startprobs.setdefault(key, 0.0)
			startprobs[key] += 0.1

		for key in CHORDS:
			transmat.setdefault(key, {})
			for val in CHORDS:
				transmat[key].setdefault(val, 0.0)
				transmat[key][val] += 0.1

		for key in CHORDS:
			emission_dict.setdefault(key, {})
			for val in range(62, 96):
				emission_dict[key].setdefault(val, 0.0)
				emission_dict[key][val] += 0.1

		model.transmat, _, _ = make_nparray_from_dict(transmat)
		model.startprob, _, _ = make_nparray_from_dict(startprobs)


		model.emissionprob, notes, chords = make_nparray_from_dict(emission_dict)
		self.chords = chords
		possible_notes = [i for i in range(62, 97)]
		possible_note_lengths =[1 for _ in range(62,97)]


		# print('s:', model.startprob)
		# print('t:', model.transmat)
		# print('e:', model.emissionprob)

		f_note_data = [f for flist in first_notes for f in flist] + possible_notes
		f_note_array = np.array(f_note_data)
		f_note_lengths = [len(f) for f in first_notes] + possible_note_lengths


		# print(f_note_array)
		minval = 62 # min(f_note_array)
		f_note_delta = np.array([(f - minval) for f in f_note_array])
		# print('fnotedelta:', f_note_delta)
		# print('fnotelengths:', f_note_lengths)

		model.fit(f_note_delta.reshape(-1, 1), lengths=f_note_lengths)

		self.clf = model

	def predict(self, notes):
		# print('notes:', notes)
		notes = np.array([(n-62) for n in notes]).reshape(-1, 1)

		# self.clf.fit(notes)

		logprob, val = self.clf.decode(notes)
		# print('val:', val)
		val = [self.chords[v] for v in val]
		# print('val2:', val)
		return logprob, val
