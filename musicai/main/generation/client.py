from musicai.main.ngrams.Ngram import *
from musicai.tests.split import *
from musicai.main.lib.input_vectors import sequence_vectors, parse_data

def gen_trigram():
	train, val, test = splitData()
	bar_sequences,chord_sequences = parse_data(train, octave=True,  generated=True)
	X_train = flatten(flatten(bar_sequences))
	Y_train = flatten(chord_sequences)
	obj = Text(list(zip(X_train,Y_train)))
	trigrams = obj.generate_trigram_sequences(30)
	print("Trigrams : ", trigrams)
	chords = trigrams[0][0:2] + tuple(i[2] for i in trigrams[1:])
	return chords
	
	
if __name__ == "__main__":
	#to test
	train, val, test = splitData()
	bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True, generated=True)
	X_train = flatten(flatten(bar_sequences))
	Y_train = flatten(flatten(chord_sequences))
	print(X_train)
	print(Y_train)
	print(list(zip(X_train,Y_train)))