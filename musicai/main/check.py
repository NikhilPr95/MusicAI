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

musicfiles = glob.glob(os.path.join(directories.PROCESSED_CHORDS, "*"))
bar_sequences, chord_sequences = parse_data(musicfiles)


# for b in bar_sequences:
# 	print(b)
#
# for bar_sequence in bar_sequences:
# 	for bar in bar_sequence:
# 		print(bar)
# 		x = bar[0]


f_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]

# for row in zip(f_notes, chord_sequences):
# 	for r in zip(row[0], row[1]):
# 		print(r)
#
print(len(bar_sequences), len(chord_sequences))
print('bs:', bar_sequences)
print('fn:', f_notes)
print('cs:', chord_sequences)

# hmm_train()
#
# x = np.array([0,1,2,1,1,1]).reshape(-1,1)
# x = np.array([0,]).reshape(-1,1)
# print(hmm_predict(x))
# for f in f_notes:
# 	print(f)

ph = PyHMM()
ph.fit(bar_sequences, chord_sequences)
ph.predict([72])