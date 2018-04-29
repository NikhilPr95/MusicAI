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
from keras.models import load_model

def lstm_predict(right_hand_notes):
	if glob.glob(os.path.join(directories.PICKLES, 'lstm.h5')):
		#lstm = load(open(os.path.join(directories.PICKLES, 'lstm.pkl'), "rb"))
		lstm = load_model(os.path.join(directories.PICKLES, 'lstm.h5'))
	else:
		bar_sequences, chord_sequences = parse_data(glob.glob(directories.PROCESSED_CHORDS+"*"), octave=True, reduce_chords=True, padding=4, padval=-1)
		# print("chords :" ,flatten(bar_sequences))
		print(chord_sequences)
		print("chords :", flatten(chord_sequences))
		
		chord_sequences = flatten(chord_sequences)
		chord_sequences = [SIMPLE_CHORDS.index(i) for i in chord_sequences]
		bar_sequences = flatten(bar_sequences)
		
		X = [[0,0,0,0]] + bar_sequences
		y = chord_sequences + [0]
		
		lstm = LSTM()
		lstm.fit(X, y)
		pred = lstm.predict([right_hand_notes[:4]])
		return SIMPLE_CHORDS[pred.argmax()]+"_M"
		
	
	right_hand_notes = np.array([right_hand_notes[:4]])
	print(right_hand_notes.shape)
	right_hand_notes = right_hand_notes.reshape(1, right_hand_notes.shape[1], 1)
	pred = lstm.predict(np.array(right_hand_notes))
	print(pred.argmax())
	print("LSTM : ", pred)
	return SIMPLE_CHORDS[pred.argmax()]+"_M"


