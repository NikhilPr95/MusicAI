from musicai.main.constants import directories
import glob, random

#from musicai.main.models.hmm import HMM
from musicai.main.models.lstm import LSTM
#from musicai.main.models.ko import KO
from musicai.main.lib.input_vectors import sequence_vectors, parse_data, get_first_note_sequences, ngram_vector
#from musicai.main.models.mlp import MLP
from musicai.tests.metrics import percentage, precision, recall, longest_bad_run, longest_good_run
#from musicai.main.models.pyhmm import PyHMM
from musicai.tests.metrics import percentage
from musicai.utils.general import *
import os,time
from musicai.main.constants.values import *
from musicai.main.constants.directories import *


def splitData():

	musicFiles_ = glob.glob(os.path.join(directories.RHYMES,"*"))
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
		obj = LSTM()
		bar_sequences, chord_sequences = parse_data(train, octave=True, reduce_chords=True, padding=4, padval=-1)
		#print("chords :" ,flatten(bar_sequences))
		print("chords :" ,flatten(chord_sequences))
		
		chord_sequences = flatten(chord_sequences)
		chord_sequences = [SIMPLE_CHORDS.index(i) for i in chord_sequences]
		numberChords = False
		if not numberChords:
			bar_sequences = flatten(bar_sequences)
			bar_sequences = [[0,0,0,0]] + bar_sequences
			print(bar_sequences)
			chord_sequences = chord_sequences + [0]
			obj.fit(bar_sequences, chord_sequences)
		else:
			chord_sequences_temp = [0,0,0,0] + chord_sequences #assuming all C chords
			chord_timesteps = [tuple(chord_sequences_temp[i:i+numberChords]) for i in range(len(chord_sequences_temp))]
			X = zip(flatten(bar_sequences), chord_timesteps)
			chord_sequences = [3] + chord_sequences[:-1]
			obj.fit(list(X), chord_sequences)
	return obj


def checkModel():
	print("Your options for the model : ")
	print(" 1) KNN and OMM")
	print(" 2) HMM ")
	print(" 3) PyHMM ")
	print(" 4) PyHMM Ngrams")
	print(" 5) RNN (Coming soon) :) ")
	# option = int(input("Enter your choice : "))
	option = 6

	dataset = splitData()
	test = dataset[2]

	bar_sequences, chord_sequences = parse_data(dataset[2], octave=True, reduce_chords=True)

	percs = []

	#print('b:', bar_sequences)
	#print('c:', chord_sequences)
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
	elif option == 2:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		actual_chords = []
		print('\n\n\n\n')
		print(len(bar_sequences))

		first_note_sequences = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
		for first_note_sequence, chord_sequence in zip(first_note_sequences, chord_sequences):
			print('f:', first_note_sequence)
			print('c:', chord_sequence)
			print(len(first_note_sequence), len(chord_sequence))
			predicted_chords.append([])
			actual_chords.append([])
			for i in range(3, len(first_note_sequence)):
				input_notes = first_note_sequence[i-3:i]
				pred_chords = obj.predict(input_notes)
				predicted_chords[-1].append(pred_chords)
				actual_chords[-1].append(chord_sequence[i])

		print('predicted:', predicted_chords)
		print('actual:', actual_chords)

		for pc, ac in zip(predicted_chords, actual_chords):
			print('pc:', pc)
			print('ac:', ac)
			for p, a in zip(pc, ac):
				print('p a', p, a)

		predicted_chords = [x[-1] for x in flatten(predicted_chords)]
		perc = percentage(flatten(actual_chords), predicted_chords)
		return perc

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

	elif option == 4:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		actual_chords = []
		ngram_chords = []
		print('\n\n\n\n')
		print(len(bar_sequences))

		first_note_sequences = [[bar[0] for bar in bar_sequence] for bar_sequence in bar_sequences]
		for first_note_sequence, chord_sequence in zip(first_note_sequences, chord_sequences):
			print('f:', first_note_sequence)
			print('c:', chord_sequence)
			print(len(first_note_sequence), len(chord_sequence))
			predicted_chords.append([])
			actual_chords.append([])
			ngram_chords.append([])
			n = obj.ngramlength
			for i in range(n, len(first_note_sequence)):
				input_notes = first_note_sequence[i-n:i]
				pred_chords = obj.predict(input_notes)
				predicted_chords[-1].append(pred_chords)
				actual_chords[-1].append(chord_sequence[i-1])
				ngram_chords[-1].append(chord_sequence[i-n:i])

		print('predicted:', predicted_chords)
		print('ngram chords:', ngram_chords)
		print('actual:', actual_chords)

		for pc, nc, ac in zip(predicted_chords, ngram_chords, actual_chords):
			for p, n, a in zip(pc, nc, ac):
				print('p n a', p, n, a)

		predicted_chords = [x[-1] for x in flatten(predicted_chords)]

		perc = percentage(flatten(actual_chords), predicted_chords)
		percs.append(perc)

	elif option == 5:
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
			predicted_chords.append(obj.predict(f_note_ngram + chord_ngram_numbers[:-1]))

		predicted_chords = [x[-1] for x in predicted_chords]
		actual_chords = [x[-1] for x in ngram_chord_sequences]

		perc = percentage(flatten(actual_chords), predicted_chords)
		percs.append(perc)

	elif option == 6:
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		bar_sequences, chord_sequences = parse_data(dataset[2], octave=True, reduce_chords=True, padding=4, padval=-1)
		
		numberChords = 4
		chord_sequences = flatten(chord_sequences)
		actual_chords = [SIMPLE_CHORDS.index(i) for i in chord_sequences]
		actual_chords = [0] + actual_chords[:-1]
		#chord_sequences_temp = [0, 0, 0, 0] + actual_chords  # assuming all C chords
		#chord_timesteps = [tuple(chord_sequences_temp[i:i + numberChords]) for i in range(len(chord_sequences_temp))]
		
		bar_sequences = flatten(bar_sequences)
		for i in range(len(bar_sequences)):
			predicted_chords.append(obj.predict([bar_sequences[i]]))

		print(predicted_chords)
		percs.append(percentage(actual_chords, predicted_chords))
		print("Test accuracy : ",percs[-1])
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
