# implements Observable Markov Model
# counts all occurrences of chords as per algorithm
# has a function, called by predict.py
# takes a chord, returns next chord in sequence
import glob
from pickle import *

# from musicai.main.lib.input_vectors import *
from musicai.main.lib.input_vectors import sequence_vectors
from musicai.main.lib.markov import get_transition_matrices


def omm_train():
	chord_sequences = []
	for file_name in glob.glob("../../data/processed_chords/*"):
		data = sequence_vectors(file_name)
		chord_sequences.append(data[1])

	return get_transition_matrices(chord_sequences)


def omm_predict(chord):
	if glob.glob("omm.pkl"):
		data = load(open("../pickles/omm.pkl", "rb"))
	else:
		data = omm_train()
		dump(data, open("../pickles/omm.pkl", "wb"))

	max = 0
	key = "X"
	for each_chord in data[1][chord]:
		if(max < data[1][chord][each_chord]):
			key = each_chord
			max = data[1][chord][each_chord]

	return key
