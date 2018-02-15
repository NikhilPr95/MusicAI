# implements Observable Markov Model
# counts all occurrences of chords as per algorithm
# has a function, called by predict.py
# takes a chord, returns next chord in sequence
import glob
from pickle import *
from musicai.main.lib.input_vectors import *
def omm_train():
	chord_sequences = []
	for file_name in glob.glob("../../data/processed_chords/*"):
		data = sequence_vectors(file_name)
		chord_sequences.append(data[1])

	start_probs = {}
	transition_probs = {}

	for chord_sequence in chord_sequences:
		if chord_sequence[0] not in start_probs:
			start_probs[chord_sequence[0]] = 0
		start_probs[chord_sequence[0]]+=1
		for i in range(len(chord_sequence)-1):
			if chord_sequence[i] not in transition_probs:
				transition_probs[chord_sequence[i]] = {}
			if chord_sequence[i + 1] not in transition_probs[chord_sequence[i]]:
				transition_probs[chord_sequence[i]][chord_sequence[i+1]] = 0
			transition_probs[chord_sequence[i]][chord_sequence[i+1]]+=1

	for chord in transition_probs:
		sum_values = sum(transition_probs[chord].values())
		for each_chord in transition_probs[chord]:
			transition_probs[chord][each_chord] = transition_probs[chord][each_chord]/sum_values

	sum_probs = sum(start_probs.values())
	for i in start_probs:
		start_probs[i] = start_probs[i]/sum_probs
	return [start_probs,transition_probs]



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
