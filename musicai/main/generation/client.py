from musicai.main.ngrams.Ngram import *
from musicai.tests.split import *
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
from musicai.main.constants.directories import *
import glob
import os.path
def gen_trigram():
	train, val, test = splitData()
	#[os.path.join(PROCESSED_CHORDS_TIME,"baba.csv.formatted")]
	bar_sequences,chord_sequences = parse_data(train, octave=True,  generated=True)
	print(len(flatten(bar_sequences)),len(flatten(bar_sequences)))
	X_train = flatten(flatten(bar_sequences))
	Y_train = flatten(chord_sequences)
	ngrams = list(zip(flatten(bar_sequences),flatten(chord_sequences)))
	ngrams = [(gram[0][i],gram[1]) for gram in ngrams for i in range(len(gram[0])) ]
	obj = Text(ngrams)
	trigrams = obj.generate_trigram_sequences(300)
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
