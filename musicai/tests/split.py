from musicai.main.models.logreg import LogReg
from musicai.main.models.svm import SVM
from sklearn import svm

from musicai.main.constants import directories
import glob, random

from musicai.main.models.hmm import HMM
from musicai.main.models.ko import KO
from musicai.main.lib.input_vectors import sequence_vectors, parse_data, get_first_note_sequences, ngram_vector, \
	create_classic_feature_matrix, create_ngram_feature_matrix, create_note_matrix
from musicai.main.models.mlp import MLP
from musicai.tests.metrics import percentage, precision, recall, longest_bad_run, longest_good_run
from musicai.main.models.pyhmm import PyHMM
from musicai.tests.metrics import percentage
from musicai.utils.general import *
import os,time
from musicai.main.constants.values import *
from musicai.main.constants.directories import *


def splitData():

	musicFiles_ = glob.glob(os.path.join(directories.PROCESSED_CHORDS,"*"))
	random.shuffle(musicFiles_)
	musicFiles = [f for f in musicFiles_ if len(open(f).readlines()) > BAR_THRESHOLD]
	length = len(musicFiles)

	trainData = musicFiles[:int(0.8*length)]+list(set(musicFiles_) - set(musicFiles))
	valData = []#musicFiles[int(0.6 * length):int(0.8 * length)]
	testData = musicFiles[int(0.8 * length):]

	print("------")
	print(len(trainData),len(testData))
	print("-----")

	return trainData, valData, testData


def fitModel(option, train):
	obj = None

	if option == 1:
		if os.path.isfile(PICKLES + "knn.pkl"):
			os.remove(PICKLES + "knn.pkl")
		if os.path.isfile(PICKLES + "omm.pkl"):
			os.remove(PICKLES + "omm.pkl")
		obj = KO()
		bar_sequences,chord_sequences = parse_data(train, padding=MAX_NOTES, octave=True, reduce_chords=True)
		obj.fit(bar_sequences,chord_sequences)

	elif option == 2:
		obj = HMM()
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 3:
		obj = PyHMM(ngrams=False)
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 4:
		obj = PyHMM(ngrams=True)
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 5:
		obj = MLP()
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 6:
		obj = SVM(data_type='first_notes')
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 7:
		obj = SVM(data_type='current_bar')
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True, padding=10, padval=-1)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 8:
		obj = LogReg()
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 9:
		obj = MLP(data_type='current_bar')
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True, padding=10, padval=-1)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 10:
		obj = LogReg(data_type='current_bar')
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True, padding=10, padval=-1)
		obj.fit(bar_sequences, chord_sequences)

	return obj


def checkModel():
	print("Your options for the model : ")
	print(" 1) KNN and OMM")
	print(" 3) PyHMM ")
	print(" 4) PyHMM Ngrams")
	print(" 5) MLP")
	print(" 6) SVM first notes")
	print(" 7) SVM current bar")
	print(" 8) Logreg first notes")
	print(" 9) MLP current bar")
	print(" 10) Logreg current bar")
	print(" 20) RNN (Coming soon) :) ")
	# option = int(input("Enter your choice : "))
	option = 1

	dataset = splitData()
	test = dataset[2]

	bar_sequences, chord_sequences = parse_data(dataset[2], octave=True, reduce_chords=True, padding=10, padval=-1)

	percs = []

	print('b:', bar_sequences)
	print('c:', chord_sequences)

	if option == 1:
		obj = fitModel(option, dataset[0])

		bar_notes, knn_labels = create_note_matrix(bar_sequences, chord_sequences, exclude=1, delta=0)
		_, omm_labels = create_note_matrix(bar_sequences, chord_sequences, exclude=1, delta=1)

		percs.append(obj.score(bar_notes, list(zip(knn_labels, omm_labels))))

	elif option in [7, 9, 10]:
		obj = fitModel(option, dataset[0])
		X, y = create_classic_feature_matrix(bar_sequences, chord_sequences)
		percs.append(obj.score(X, y))

	elif option in [3, 4, 5, 6, 8]:
		obj = fitModel(option, dataset[0])
		X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, n=obj.ngramlength, chords=False)
		percs.append(obj.score(X, y))

	return percs


if __name__ == "__main__":
	results = checkModel()
	print('results:', results)
