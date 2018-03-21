from musicai.main.ngrams.Ngram import *
#from musicai.tests.split import *
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
from musicai.main.constants.directories import *
import glob
def gen_trigram():
	#train, val, test = splitData()
	#bar_sequences,chord_sequences = parse_data(train, octave=True, reduce_chords=True, generated=True)
	bar_sequences,chord_sequences = parse_data(glob.glob(PROCESSED_CHORDS_TIME + "titanic.csv.formatted"), octave=True, reduce_chords=True, generated=True)
	X_train = flatten(flatten(bar_sequences))
	obj = Text(X_train)
	trigrams = obj.generate_trigram_sequences(3000)
	#print("Trigrams : ", trigrams)
	chords = trigrams[0][0:2] + tuple(i[2] for i in trigrams[1:])
	return chords
	
