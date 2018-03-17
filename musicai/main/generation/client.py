from musicai.main.ngrams.Ngram import *
from musicai.tests.split import *
from musicai.main.lib.input_vectors import sequence_vectors, parse_data

def gen_bigram():
	train, val, test = splitData()
	bar_sequences,chord_sequences = parse_data(train, padding=MAX_NOTES, octave=True, reduce_chords=True)
	obj = Text(flatten(flatten(bar_sequences)))
	return obj.generate_bigram_sequences(30)



