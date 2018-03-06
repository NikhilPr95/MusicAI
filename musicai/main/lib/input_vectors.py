import csv
import glob

import os

from musicai.main.constants import directories


def sequence_vectors(csvfilepath, padding = 0):	# padding is the len of the vector required
    def getdata(csvfile, data, labels, maxlen):
        rows = csv.reader(open(csvfile, "r"))

        for row in rows:
            left_note_inputs = row[1].split('-')
            bar = [int(note_val.split('|')[0]) for note_val in left_note_inputs if note_val.split('|')[1] != '0']

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
            #print(csvfilepath)
            if csvfile.endswith('csv.formatted'):
                maxlen = getdata(csvfilepath+'/'+csvfile, data, labels, maxlen)
    if padding:
        for bar in data:
            if len(bar) < padding:
                bar.extend([0]*(padding-len(bar)))
            else:
                bar = bar[:padding]
    return data, labels


def parse_data(csvfilepath):
	"""
	Parses csvs and returns bar and chord seqeunces
	Args:
		csvfilepath: Path to music data csv

	Returns:
	Bar and chord sequences
	"""
	bar_sequences = []
	chord_sequences = []
	for csvfile in glob.glob(os.path.join(directories.PROCESSED_CHORDS, '*')):
		data = sequence_vectors(csvfile)
		bar_sequences.append(data[0])
		chord_sequences.append(data[1])

	return bar_sequences, chord_sequences
