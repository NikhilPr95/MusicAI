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
bar_sequences, chord_sequences = parse_data(directories.PROCESSED_CHORDS)


for b in bar_sequences:
	print(b)

# for bar_sequence in bar_sequences:
# 	for bar in bar_sequence:
# 		print(bar)
# 		x = bar[0]
#

# f_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
#
# for row in zip(f_notes, chord_sequences):
# 	for r in zip(row[0], row[1]):
# 		print(r)
#
# print(bar_sequences)
# print(len(bar_sequences))
# print(len(bar_sequences[0]))

# print(f_notes)
# for f in f_notes:
# 	print(f)

#
#
# print(len(chord_sequences), len(chord_sequences[0]))
# print(len(f_notes), len(f_notes[0]))

# for bar in bar_sequences:
# 	print(bar)
#
# for c in chord_sequences:
# 	print(c)
#
# for f in f_notes:
# 	print(f)
#
# for row in zip(f_notes, chord_sequences):
# 	print(row)
# # 	print(len(row[0]), len(row[1]))
# f= [f for f_list in f_notes for f in f_list]
# c = [c for c_list in chord_sequences for c in c_list]
# print(f)
# print(c)
# print(len(f), len(c))

# print(np.concatenate([f for f in f_notes]))

hmm_train()

x = np.array([0,1,2,1,1,1]).reshape(-1,1)
x = np.array([0,]).reshape(-1,1)
print(hmm_predict(x))
# for f in f_notes:
# 	print(f)