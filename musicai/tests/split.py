from musicai.main.constants import directories
import glob, random

from musicai.main.models.hmm import HMM
from musicai.main.models.ko import KO
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
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
		bar_sequences,chord_sequences = parse_data(train, padding=MAX_NOTES)
		obj.fit(bar_sequences,chord_sequences)

	elif option == 2:
		obj = HMM()
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train)
		obj.fit(bar_sequences, chord_sequences)

	elif option == 3:
		obj = PyHMM()
		print('train:', train)
		bar_sequences, chord_sequences = parse_data(train, padding=3)
		obj.fit(bar_sequences, chord_sequences)

	return obj


def checkModel():
	print("Your options for the model : ")
	print(" 1) KNN and OMM")
	print(" 2) HMM ")
	print(" 3) RNN (Coming soon) :) ")
	option = int(input("Enter your choice : "))

	dataset = splitData()
	test = dataset[2]

	#songs_bar_sequences, songs_chord_sequences = parse_data(test,padding=MAX_NOTES)


	songs_bar_sequences, songs_chord_sequences = parse_data(test, padding=3)

	if option == 1:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_knn, predicted_omm = [], []
		for bar_sequences in songs_bar_sequences:
			predicted_knn.append([])
			predicted_omm.append([])
			for bar_sequence in bar_sequences[:-1]: #this is required as the previous bar is sent to omm, hence we need to remove the last bar.
				result = obj.predict(bar_sequence)
				predicted_knn[-1].append(result[0])
				predicted_omm[-1].append(result[1])

		#the predicted and song_chord_sequences is for each song, so we have to iterate through the song and combine all values to one list
		print("Length of test : "+str(len(flatten(predicted_knn))))
		perc_knn = percentage(flatten(songs_chord_sequences),flatten(predicted_knn))
		omm_actual , omm_predicted = [chord_sequence for chord_sequences in songs_chord_sequences for chord_sequence in chord_sequences[1:]], flatten(predicted_omm)
		perc_omm = percentage(omm_actual, omm_predicted)
		precision1 = precision(omm_actual, omm_predicted)
		recall1 = recall(omm_actual, omm_predicted)
		longest_good_run1 = longest_good_run(omm_actual, omm_predicted)
		longest_bad_run1 = longest_bad_run(omm_actual, omm_predicted)
		return perc_knn, perc_omm,  precision1, recall1, longest_good_run1, longest_bad_run1
	elif option == 2:
		print(dataset)
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		for bar_sequences in songs_bar_sequences:
			predicted_chords.append([])
			for bar_sequence in bar_sequences:
				logval, pred_chords = obj.predict(bar_sequence)
				predicted_chords[-1].append(pred_chords)

		print('songc:', songs_chord_sequences)
		print('predc:', predicted_chords)
		print('lens:', len(songs_chord_sequences), len(predicted_chords), [len(s) for s in songs_chord_sequences],
		[len(c) for c in predicted_chords])

		# last_notes = [[bar[-1] for bar in chord_sequence] for chord_sequence in predicted_chords]
		predicted_chords = [x[-1] for x in flatten(predicted_chords)]
		print('\n\npc:', predicted_chords)
		print('s:', flatten(songs_chord_sequences))
		perc = percentage(flatten(songs_chord_sequences), predicted_chords)
		precision1 = precision(flatten(songs_chord_sequences), predicted_chords)
		recall1 = recall(flatten(songs_chord_sequences), predicted_chords)
		longest_good_run1 = longest_good_run(flatten(songs_chord_sequences), predicted_chords)
		longest_bad_run1 = longest_bad_run(flatten(songs_chord_sequences), predicted_chords)
		return perc, precision1, recall1, longest_good_run1, longest_bad_run1

	elif option == 3:
		print('train:', dataset[0])
		obj = fitModel(option, dataset[0])
		predicted_chords = []
		for bar_sequences in songs_bar_sequences:
			predicted_chords.append([])
			for bar_sequence in bar_sequences:
				pred_chords = obj.predict(bar_sequence)
				predicted_chords[-1].append(pred_chords)

		print('predicted:', predicted_chords)
		for pc in predicted_chords:
			print('pc:', pc)

		predicted_chords = [x[-1] for x in flatten(predicted_chords)]
		perc = percentage(flatten(songs_chord_sequences), predicted_chords)
		return perc

	elif option == 4:
		print("Haha! You thought null pointer, didn't you? \n Coming soon!\n\n\n\n The RNN, not the null pointer")


if __name__ == "__main__":
	# print(testModel())
	results = checkModel()
	print("Accuracy of KNN : ", str(results[0]))
	print("Accuracy of OMM : ", str(results[1]))
	print("Precision : ", str(results[2]))
	print("Recall : ", str(results[3]))
	print("Longest good run : ", str(results[4]))
	print("Longest bad run : ", str(results[5]))
