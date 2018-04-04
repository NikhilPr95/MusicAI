# implements Observable Markov Model
# counts all occurrences of chords as per algorithm
# has a function, called by predict.py
# takes a chord, returns next chord in sequence
import glob
from collections import OrderedDict
from pickle import *

import os

import numpy as np
from hmmlearn import hmm
from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
from musicai.utils.general import flatten, make_nparray_from_dict


def transition_matrices(sequences):
	start_probs = {}
	transition_probs = {}

	# print('s:', sequences)
	seq = flatten(sequences)
	for state in set(seq):
		transition_probs.setdefault(state, {})
		for label in set(seq):
			transition_probs[state].setdefault(label, 0.0)

	for state in set(seq):
		start_probs.setdefault(state, 0.0)

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
			transition_probs[state][each_chord] = transition_probs[state][each_chord] / sum_values \
				if sum_values != 0 else 0

	sum_probs = sum(start_probs.values())
	for i in start_probs:
		start_probs[i] = start_probs[i] / sum_probs
	return [start_probs, transition_probs]


def emission_matrix(state_sequence, labels):
	emission_probs = dict()

	for state in set(state_sequence):
		emission_probs.setdefault(state, {})
		for label in set(labels):
			emission_probs[state].setdefault(label, 0)

	for state, label in zip(state_sequence, labels):
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
	bar_sequences, chord_sequences = parse_data(glob.glob(directories.PROCESSED_CHORDS))

	# print(bar_sequences)
	first_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
	all_chords = flatten(chord_sequences)
	all_notes = flatten(first_notes)
	num_chords = len(set(all_chords))

	model = hmm.MultinomialHMM(num_chords)

	model.startprob, model.transmat = transition_matrices(first_notes)
	emission_dict = emission_matrix(all_notes, all_chords)

	model.emissionprob, notes, chords = make_nparray_from_dict(emission_dict)

	possible_notes = [i for i in range(62, 96)]
	possible_note_lengths = [1 for _ in range(62, 96)]

	f_note_data = [f for flist in first_notes for f in flist] + possible_notes
	f_note_array = np.array(f_note_data)
	f_note_lengths = [len(f) for f in first_notes] + possible_note_lengths

	# print('fn:', len(f_note_array))
	# print('sum:', sum(f_note_lengths))
	# print(f_note_array)
	minval = 62  # min(f_note_array)
	f_note_delta = np.array([(f - minval) for f in f_note_array])
	# print('fnotedelta:', f_note_delta)
	# print('fnotelengths:', f_note_lengths)

	model.fit(f_note_delta.reshape(-1, 1), lengths=f_note_lengths)

	# self.clf = model

	return model


def hmm_predict(notes):
	# if glob.glob(os.path.join(directories.PICKLES, 'hmm.pkl')):
	# 	model = load(open(os.path.join(directories.PICKLES, 'hmm.pkl'), "rb"))
	# else:
	model = hmm_train()
	dump(model, open(os.path.join(directories.PICKLES, 'hmm.pkl'), "wb"))

	logprob, val = model.decode(notes)
	return logprob, val
