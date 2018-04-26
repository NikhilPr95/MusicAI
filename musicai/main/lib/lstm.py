import numpy as np
from musicai.main.constants import directories
from sklearn.neighbors import KNeighborsClassifier
from musicai.main.lib.input_vectors import sequence_vectors,parse_data
import glob
from pickle import load, dump
from musicai.main.constants.directories import *
from musicai.main.models.lstm import *
from musicai.utils.general import *
from musicai.main.constants.values import *


def lstm_predict(right_hand_notes):
	if glob.glob(os.path.join(directories.PICKLES, 'lstm.pkl')):
		lstm = load(open(os.path.join(directories.PICKLES, 'lstm.pkl'), "rb"))
	else:
		data = sequence_vectors(directories.PROCESSED_CHORDS, padding=4)
		
		bar_sequences, chord_sequences = parse_data(data, octave=True, reduce_chords=True, padding=4, padval=-1)
		# print("chords :" ,flatten(bar_sequences))
		print("chords :", flatten(chord_sequences))
		
		chord_sequences = flatten(chord_sequences)
		chord_sequences = [SIMPLE_CHORDS.index(i) for i in chord_sequences]
		bar_sequences = flatten(bar_sequences)
		
		X = [[0,0,0,0]] + bar_sequences
		y = chord_sequences + [0]
		
		lstm = LSTM()
		lstm.fit(X, y)
		
	outp = lstm.predict([right_hand_notes])
	print("LSTM : ", outp)
	return outp


