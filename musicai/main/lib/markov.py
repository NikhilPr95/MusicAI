# implements Observable Markov Model
# counts all occurrences of chords as per algorithm
# has a function, called by predict.py
# takes a chord, returns next chord in sequence
import glob
from pickle import *

# from musicai.main.lib.input_vectors import *
from musicai.main.lib.input_vectors import sequence_vectors


def get_transition_matrices(sequences):
	start_probs = {}
	transition_probs = {}

	for sequence in sequences:
		if sequence[0] not in start_probs:
			start_probs[sequence[0]] = 0
		start_probs[sequence[0]] += 1
		for i in range(len(sequence) - 1):
			if sequence[i] not in transition_probs:
				transition_probs[sequence[i]] = {}
			if sequence[i + 1] not in transition_probs[sequence[i]]:
				transition_probs[sequence[i]][sequence[i + 1]] = 0
			transition_probs[sequence[i]][sequence[i + 1]] += 1

	for state in transition_probs:
		sum_values = sum(transition_probs[state].values())
		for each_chord in transition_probs[state]:
			transition_probs[state][each_chord] = transition_probs[state][each_chord] / sum_values

	sum_probs = sum(start_probs.values())
	for i in start_probs:
		start_probs[i] = start_probs[i] / sum_probs
	return [start_probs, transition_probs]


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
