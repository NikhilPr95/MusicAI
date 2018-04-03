import glob
import random

from musicai.main.constants import directories
from musicai.main.constants.directories import *
from musicai.main.constants.values import *
from musicai.main.lib.input_vectors import parse_data, \
	create_classic_feature_matrix, create_ngram_feature_matrix, create_note_matrix
from musicai.main.models.ko import KO
from musicai.main.models.logreg import LogReg
from musicai.main.models.mlp import MLP
from musicai.main.models.pyhmm import PyHMM
from musicai.main.models.svm import SVM


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


def fitModel(train, model=None, data_type=None, padding=None, padval=0):
	if model == KO:
		if os.path.isfile(PICKLES + "knn.pkl"):
			os.remove(PICKLES + "knn.pkl")
		if os.path.isfile(PICKLES + "omm.pkl"):
			os.remove(PICKLES + "omm.pkl")
		padding = MAX_NOTES if padding is None else padding

	if data_type == 'current_bar':
		padding = 10 if padding is None else padding
		padval = -1 if padval is None else padval

	obj = model(data_type=data_type)
	bar_sequences_train, chord_sequences_train = parse_data(train, padding=padding, padval=padval)
	obj.fit(bar_sequences_train, chord_sequences_train)

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

	option = 9

	dataset = splitData()
	test = dataset[2]

	bar_sequences, chord_sequences = parse_data(test, octave=True, reduce_chords=True, padding=10, padval=-1)

	percs = []

	print('b:', bar_sequences)
	print('c:', chord_sequences)

	if option == 1:
		model = KO
		obj = fitModel(dataset[0], model=model)

		bar_sequences, chord_sequences = parse_data(test, octave=True, reduce_chords=True, padding=MAX_NOTES, padval=0)

		percs.append(obj.score(bar_sequences, chord_sequences))
	elif option in [7, 9, 10]:
		model = None
		if option == 7:
			model = SVM

		elif option == 9:
			model = MLP

		elif option == 10:
			model = LogReg

		data_type = 'current_bar'

		obj = fitModel(dataset[0], model=model, data_type=data_type)

		percs.append(obj.score(bar_sequences, chord_sequences))

	elif option in [3, 4, 5, 6, 8]:
		model, data_type = None, None
		if option == 3:
			model = PyHMM
			data_type = 'sequence'

		elif option == 4:
			model = PyHMM
			data_type = 'ngram'

		elif option == 5:
			model = MLP
			data_type = 'first_notes'

		elif option == 6:
			model = SVM
			data_type = 'first_notes'

		elif option == 8:
			model = LogReg
			data_type = 'first_notes'

		obj = fitModel(dataset[0], model=model, data_type=data_type)

		percs.append(obj.score(bar_sequences, chord_sequences))

	return percs


if __name__ == "__main__":
	results = checkModel()
	print('results:', results)
