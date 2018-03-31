from musicai.main.models.logreg import LogReg
from musicai.main.models.svm import SVM
from sklearn import svm

from musicai.main.constants import directories
import glob, random

from musicai.main.models.hmm import HMM
from musicai.main.models.ko import KO
from musicai.main.lib.input_vectors import sequence_vectors, parse_data, get_first_note_sequences, ngram_vector, \
	create_classic_feature_matrix
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
	option = 4

	dataset = splitData()
	test = dataset[2]

	bar_sequences, chord_sequences = parse_data(dataset[2], octave=True, reduce_chords=True, padding=10, padval=-1)

	percs = []

	print('b:', bar_sequences)
	print('c:', chord_sequences)

	if option == 1:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_knn, predicted_omm = [], []
		for bars in bar_sequences:
			predicted_knn.append([])
			predicted_omm.append([])
			for bar in bars[:-1]: #this is required as the previous bar is sent to omm, hence we need to remove the last bar.
				result = obj.predict(bar)
				predicted_knn[-1].append(result[0])
				predicted_omm[-1].append(result[1])

		for pk, po in zip(predicted_knn, predicted_omm):
			for p, o in zip(pk, po):
				print('p o:', p, o)

		#the predicted and song_chord_sequences is for each song, so we have to iterate through the song and combine all values to one list
		print("Length of test : "+str(len(flatten(predicted_knn))))
		perc_knn = percentage(flatten(chord_sequences),flatten(predicted_knn))
		omm_actual , omm_predicted = [chord for chords in chord_sequences for chord in chords[1:]], flatten(predicted_omm)
		perc_omm = percentage(omm_actual, omm_predicted)
		precision1 = precision(omm_actual, omm_predicted)
		recall1 = recall(omm_actual, omm_predicted)
		longest_good_run1 = longest_good_run(omm_actual, omm_predicted)
		longest_bad_run1 = longest_bad_run(omm_actual, omm_predicted)
		return perc_knn, perc_omm,  precision1, recall1, longest_good_run1, longest_bad_run1

	elif option == 3:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		print('\n\n\n\n')
		print(len(bar_sequences))
		n = obj.ngramlength

		first_note_sequences = get_first_note_sequences(bar_sequences)

		first_note_sequence_ngrams, \
		chord_sequence_ngrams = ngram_vector(first_note_sequences, n), ngram_vector(chord_sequences, n)

		ngram_chord_sequences = flatten(chord_sequence_ngrams)
		ngram_f_note_sequences = flatten(first_note_sequence_ngrams)

		for f_note_ngram in ngram_f_note_sequences:
			predicted_chords.append(obj.predict(f_note_ngram))

		predicted_chords = [x[-1] for x in predicted_chords]
		actual_chords = [x[-1] for x in ngram_chord_sequences]

		perc = percentage(flatten(actual_chords), predicted_chords)
		percs.append(perc)

	elif option in [7, 9, 10]:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		X, y = create_classic_feature_matrix(bar_sequences, chord_sequences)
		percs.append(obj.score(X, y))

	elif option in [4, 5, 6, 8]:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		print('\n\n\n\n')
		print(len(bar_sequences))
		n = obj.ngramlength

		first_note_sequences = get_first_note_sequences(bar_sequences)

		first_note_sequence_ngrams, \
		chord_sequence_ngrams = ngram_vector(first_note_sequences, n), ngram_vector(chord_sequences, n)

		ngram_chord_sequences = flatten(chord_sequence_ngrams)
		ngram_f_note_sequences = flatten(first_note_sequence_ngrams)

		for f_note_ngram, chord_ngram in zip(ngram_f_note_sequences, ngram_chord_sequences):
			chord_ngram_numbers = [SIMPLE_CHORDS.index(c) for c in chord_ngram]
			# predicted_chords.append(obj.predict(f_note_ngram + chord_ngram_numbers[:-1]))
			predicted_chords.append(obj.predict(f_note_ngram))

		predicted_chords = [x[-1] for x in predicted_chords]
		actual_chords = [x[-1] for x in ngram_chord_sequences]

		perc = percentage(flatten(actual_chords), predicted_chords)
		percs.append(perc)

	elif option == 8:
		print("Haha! You thought null pointer, didn't you? \n Coming soon!\n\n\n\n The RNN, not the null pointer")

	return percs


if __name__ == "__main__":
	# print(testModel())
	results = checkModel()
	print('results:', results)
	# print("Accuracy of KNN : ", str(results[0]))
	# print("Accuracy of OMM : ", str(results[1]))
	# print("Precision : ", str(results[2]))
	# print("Recall : ", str(results[3]))
	# print("Longest good run : ", str(results[4]))
	# print("Longest bad run : ", str(results[5]))
