# implements Observable Markov Model
# counts all occurrences of chords as per algorithm
# has a function, called by predict.py
# takes a chord, returns next chord in sequence
import glob
from pickle import *

import os

from hmmlearn import hmm
from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors


def transition_matrices(sequences):
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


def emission_matrix(state_sequence, labels):
	emission_probs = dict()

	for state, label in zip(state_sequence, labels):
		emission_probs.setdefault(state, {})
		emission_probs[state].setdefault(label, 0)
		emission_probs[state][label] += 1

	emission_probs = {
		state: {label: count/sum(emission_probs[state].values()) for label, count in emission_probs[state].items()}
		for state in emission_probs}

	return emission_probs


def omm_train():
	chord_sequences = []
	for file_name in glob.glob("musicai/data/processed_chords/*"):
		data = sequence_vectors(file_name)
		chord_sequences.append(data[1])

	return transition_matrices(chord_sequences)


def omm_predict(chord):
	if glob.glob("musicai/main/pickles/omm.pkl"):  # change to os.path.join(directories.MAIN, 'pickles', omm.pkl')
		data = load(open("musicai/main/pickles/omm.pkl", "rb"))
	else:
		data = omm_train()
		dump(data, open("musicai/main/pickles/omm.pkl", "wb"))

	max = 0
	key = "X"
	for each_chord in data[1][chord]:
		if(max < data[1][chord][each_chord]):
			key = each_chord
			max = data[1][chord][each_chord]

	return key


def hmm_train():
	data, chord_sequences = sequence_vectors(directories.PROCESSED_CHORDS)
	first_notes = [d[0] for d in data]

	model = hmm.MultinomialHMM(len(set(chord_sequences)))

	model.startprob, model.transmat = transition_matrices(first_notes)
	model.emissionprob = emission_matrix(first_notes, chord_sequences)
	model.fit(first_notes)

	return model


def hmm_predict(note):
	if glob.glob(os.path.join(directories.PICKLES, 'hmm.pkl')):
		model = load(open(os.path.join(directories.PICKLES, 'hmm.pkl', "rb")))
	else:
		model = hmm_train()
		dump(model, open(os.path.join(directories.PICKLES, 'hmm.pkl', "wb")))

	logprob, val = model.decode(note)
	return logprob, val
