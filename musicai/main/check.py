import glob

import numpy as np
from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
# from musicai.main.lib.markov import transition_matrices
#
# file = directories.PROCESSED_CHORDS
#
# data, labels = sequence_vectors(file)
# # data, labels = sequence_vectors("../data/processed_chords/happy_birthday.csv.formatted")
#
# for row in zip(data, labels):
#     print(row)
#
# print(labels)
# t = transition_matrices(labels)
from musicai.main.lib.markov import hmm_train, hmm_predict

#hmm_train()
import os
from musicai.main.models.pyhmm import PyHMM
from musicai.utils.general import flatten

musicfiles = glob.glob(os.path.join(directories.PROCESSED_CHORDS, "*"))
bar_sequences, chord_sequences = parse_data(musicfiles, reduce_chords=True, octave=True)
#
#
# # for b in bar_sequences:
# # 	print(b)
# #
# # for bar_sequence in bar_sequences:
# # 	for bar in bar_sequence:
# # 		print(bar)
# # 		x = bar[0]
#
#
n = 4

first_note_sequences = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
first_note_sequence_ngrams, chord_sequence_ngrams = [], []
for first_note_sequence, chord_sequence in zip(first_note_sequences, chord_sequences):
	first_note_sequence_ngrams.append([])
	chord_sequence_ngrams.append([])
	for i in range(n, len(first_note_sequence) + 1):
		first_note_sequence_ngrams[-1].append(first_note_sequence[i - n: i])
		chord_sequence_ngrams[-1].append(chord_sequence[i - n: i])

for row in zip(first_note_sequences, first_note_sequence_ngrams, chord_sequences, chord_sequence_ngrams):
	print(row[0], row[1])
	print(row[2], row[3])

print('csn:', chord_sequence_ngrams)
print('cs:', chord_sequences)

print('fsn:', first_note_sequence_ngrams)
print('fs:', first_note_sequences)

ngram_f_note_sequences = flatten(first_note_sequence_ngrams)
ngram_chord_sequences = flatten(chord_sequence_ngrams)

print('fsnf:', ngram_f_note_sequences)
print('csnf:', ngram_chord_sequences)
# #
# print(len(bar_sequences), len(chord_sequences))
# print('bs:', bar_sequences)
# print('fn:', f_notes)
# print('cs:', chord_sequences)
#
# # hmm_train()
# #
# # x = np.array([0,1,2,1,1,1]).reshape(-1,1)
# # x = np.array([0,]).reshape(-1,1)
# # print(hmm_predict(x))
# # for f in f_notes:
# # 	print(f)
#
# ph = PyHMM()
# ph.fit(bar_sequences, chord_sequences)
# ph.predict([72])
from musicai.utils.chords import count_chords

# x = (count_chords())
# print(x)
# print(sum(x.values()))