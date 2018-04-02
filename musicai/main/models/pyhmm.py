from musicai.main.constants.values import CHORDS, NOTES, SIMPLE_CHORDS
from musicai.main.lib.input_vectors import ngram_vector, get_first_note_sequences
from musicai.main.lib.markov import transition_matrices, emission_matrix
from musicai.main.models.base import Base
from musicai.utils.general import flatten

import py_hmm


class PyHMM(Base):
	def __init__(self, ngrams=False):
		Base.__init__(self)
		self.clf = None
		self.chords = None
		self.ngrams = ngrams
		self.ngramlength = 4

	@staticmethod
	def smooth(startprob, transmat, emission_dict):
		for key in SIMPLE_CHORDS:
			startprob.setdefault(key, 0.0)
			startprob[key] += 0.01

		for key in SIMPLE_CHORDS:
			transmat.setdefault(key, {})
			for val in SIMPLE_CHORDS:
				transmat[key].setdefault(val, 0.0)
				transmat[key][val] += 0.01

		for key in SIMPLE_CHORDS:
			emission_dict.setdefault(key, {})
			for val in range(0, 13):
				emission_dict[key].setdefault(val, 0.0)
				emission_dict[key][val] += 0.01

		return startprob, transmat, emission_dict

	def fit(self, bar_sequences, chord_sequences):
		first_note_sequences = get_first_note_sequences(bar_sequences)
		n = self.ngramlength
		if self.ngrams:
			first_note_sequence_ngrams, \
			chord_sequence_ngrams = ngram_vector(first_note_sequences, n), ngram_vector(chord_sequences, n)

			ngram_chord_sequences = flatten(chord_sequence_ngrams)

			all_ngram_chords = flatten(ngram_chord_sequences)
			all_ngram_notes = flatten(flatten(first_note_sequence_ngrams))

			startprob, transmat = transition_matrices(ngram_chord_sequences)
			emission_dict = emission_matrix(all_ngram_chords, all_ngram_notes)

		else:
			all_chords = flatten(chord_sequences)
			all_notes = flatten(first_note_sequences)

			startprob, transmat = transition_matrices(chord_sequences)
			emission_dict = emission_matrix(all_chords, all_notes)

		startprob, transmat, emission_dict = PyHMM.smooth(startprob, transmat, emission_dict)

		model = py_hmm.Model(SIMPLE_CHORDS, NOTES, startprob, transmat, emission_dict)

		print('n:', NOTES)
		print('c:', SIMPLE_CHORDS)
		print('sP:', startprob)
		print('tm:', transmat)
		print('em:', emission_dict)
		self.clf = model

	def predict(self, notes):
		# print('\n\n\n\n\n')
		print('notes:', notes)
		# print('sy:', self.clf._symbols)
		prob = self.clf.evaluate(notes)
		output_chords = self.clf.decode(notes)
		# print('st:', self.clf._states)
		# print('tr:', self.clf.trans_prob)
		# print('em:', self.clf.emit_prob)

		print('OC:', output_chords)
		print('prob:', prob)

		return SIMPLE_CHORDS.index(output_chords[-1])

	def score(self, data, labels):
		return sum([1 if self.predict(d) == l else 0 for (d, l) in zip(data, labels)]) / len(data)
