import csv
import glob

import os

from musicai.main.constants import directories


def sequence_vectors(csvfilepath, padding = 0):	# padding is the len of the vector required
	def getdata(csvfile, data, labels, maxlen):
		rows = csv.reader(open(csvfile, "r"))

		for row in rows:
			right_note_inputs = row[0].split('-')
			if right_note_inputs[0] != '':
				bar = [int(note_val.split('|')[0]) for note_val in right_note_inputs if note_val.split('|')[1] != '0']

				if len(bar):
					if len(bar) > maxlen:
						maxlen = len(bar)

					data.append(bar)
					labels.append(row[2])

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
				bar.extend([0]*(padding-len(bar)))
			else:
				bar = bar[:padding]
	return data, labels


# TODO: change this to take in list of csvs
def parse_data(csvfiles, padding=0):
	"""
	Parses csvs and returns bar and chord seqeunces
	Args:
		csvfilepath: Path to music data csv

	Returns:
	Bar and chord sequences
	"""
	bar_sequences = []
	chord_sequences = []
	# for csvfile in glob.glob(os.path.join(directories.PROCESSED_CHORDS, '*')):
	print(len(csvfiles))
	for csvfile in csvfiles:
		print('file:', csvfile)
		data = sequence_vectors(csvfile, padding)
		bar_sequences.append(data[0])
		chord_sequences.append(data[1])

	return bar_sequences, chord_sequences
