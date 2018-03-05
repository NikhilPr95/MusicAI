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
from musicai.main.lib.markov import hmm_train


#hmm_train()
bar_sequences, chord_sequences = parse_data(directories.PROCESSED_CHORDS)
f_notes = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]


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

# hmm_train()
for f in f_notes:
	print(f)