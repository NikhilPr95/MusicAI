import csv
import glob

import os

from musicai.main.constants import directories
from musicai.utils.chords import reduce


def sequence_vectors(csvfilepath, padding = None, chords=False, octave=False, reduce_chords=False):	# padding is the len of the vector required
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
					bar.extend([0]*(padding-len(bar)))
			else:
				del bar[padding:]
	return data, labels


def parse_data(csvfilepaths, padding=None, chords=False, octave=False, reduce_chords=False):
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
		data = sequence_vectors(csvfile, padding, chords, octave, reduce_chords)
		bar_sequences.append(data[0])
		chord_sequences.append(data[1])

	return bar_sequences, chord_sequences
