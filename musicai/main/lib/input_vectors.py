import csv
import glob

import os

from musicai.main.constants import directories
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.utils.chords import reduce
from musicai.utils.general import flatten


def sequence_vectors(csvfilepath, padding = None, chords=False, octave=False, reduce_chords=False, padval=0): # padding is the len of the vector required
	def getdata(csvfile, data, labels, maxlen):
		rows = csv.reader(open(csvfile, "r"))

		for row in rows:
			right_note_inputs = row[0].split('-')
			if right_note_inputs[0] != '':
				bar = [int(note_val.split('|')[0]) for note_val in right_note_inputs if note_val.split('|')[1] != '0']
				if octave:
					bar = [b % 12 for b in bar]

				if len(bar):
					if len(bar) > maxlen:
						maxlen = len(bar)

					data.append(bar)
					label = row[2]
					if reduce_chords:
						labels.append(reduce(label))
					else:
						labels.append(label)

		return maxlen

	data = []
	labels = []
	maxlen = 0

	if os.path.isfile(csvfilepath):
		maxlen = getdata(csvfilepath, data, labels, maxlen)

	elif os.path.isdir(csvfilepath):
		for csvfile in os.listdir(csvfilepath):
			if csvfile.endswith('csv.formatted'):
				maxlen = getdata(csvfilepath+'/'+csvfile, data, labels, maxlen)
	if padding:
		for bar in data:
			if len(bar) < padding:
				if chords:
					bar.extend([62]*(padding-len(bar)))
				else:
					bar.extend([padval]*(padding-len(bar)))
			else:
				del bar[padding:]
	return data, labels


def ngram_vector(sequences, n):
	sequences_ngrams = []
	for sequence in sequences:
		sequences_ngrams.append([])
		for i in range(n, len(sequence) + 1):
			sequences_ngrams[-1].append(sequence[i - n: i])

	return sequences_ngrams


def create_ngram_feature_matrix(bar_sequences, chord_sequences, n, chords=False):
	first_note_sequence_ngrams, \
	chord_sequence_ngrams = ngram_vector(get_first_note_sequences(bar_sequences), n), ngram_vector(chord_sequences, n)

	ngram_chord_sequences = flatten(chord_sequence_ngrams)
	ngram_f_note_sequences = flatten(first_note_sequence_ngrams)

	X, y = [], []
	for f_note_ngram, chord_ngram in zip(ngram_f_note_sequences, ngram_chord_sequences):
		chord_ngram_numbers = [SIMPLE_CHORDS.index(c) for c in chord_ngram]
		if chords:
			X.append(f_note_ngram + chord_ngram_numbers[:-1])
		else:
			X.append(f_note_ngram)
		y.append(chord_ngram_numbers[-1])

	return X, y


def create_classic_feature_matrix(bar_sequences, chord_sequences):
	X, y = [], []
	for bar_sequence, chord_sequence in zip(bar_sequences, chord_sequences):
		for bar, chord in zip(bar_sequence, chord_sequence):
			X.append(bar)
			y.append(chord)

	return X, y


def create_note_matrix(bar_sequences, chord_sequences, exclude=1, delta=0):
	X, y = [], []
	for bars, chords in zip(bar_sequences, chord_sequences):
		for i in range(len(chords) - exclude):
			X.append(bars[i])
			y.append(chords[i + delta])

	return X, y


def get_first_note_sequences(bar_sequences):
	return [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]


def parse_data(csvfilepaths, octave=True, reduce_chords=True, chords=False, padding=None, padval=0):
	"""
	Parses csvs and returns bar and chord seqeunces
	Args:
		csvfilepath: Path to music data csv

	Returns:
	Bar and chord sequences
	"""
	bar_sequences = []
	chord_sequences = []
	for csvfile in csvfilepaths:
		print('file:', csvfile)
		data = sequence_vectors(csvfile, padding, chords, octave, reduce_chords, padval)
		bar_sequences.append(data[0])
		chord_sequences.append(data[1])

	return bar_sequences, chord_sequences
