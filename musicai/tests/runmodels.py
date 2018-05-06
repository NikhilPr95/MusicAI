import glob
import random

import yaml
from musicai.main.constants import directories
from musicai.main.constants.directories import *
from musicai.main.constants.values import *
from musicai.main.lib.input_vectors import parse_data
from musicai.main.models.ko import KO
from musicai.main.models.logreg import LogReg
from musicai.main.models.mlp import MLP
from musicai.main.models.pyhmm import PyHMM
from musicai.main.models.svm import SVM

model_class = {'KO': KO, 'SVM': SVM, 'MLP': MLP, 'LogReg': LogReg, 'PyHMM': PyHMM}


def num_bars(files):
	return sum([len(open(f).readlines()) for f in files])


def randomSplit(directory):
	musicFiles_ = glob.glob(os.path.join(directory, "*"))
	random.shuffle(musicFiles_)
	musicFiles = [f for f in musicFiles_ if len(open(f).readlines()) > BAR_THRESHOLD]
	length = len(musicFiles)

	trainData = musicFiles[:int(0.8 * length)] + list(set(musicFiles_) - set(musicFiles))
	train_bars = sum([len(open(f).readlines()) for f in trainData])
	testData = musicFiles[int(0.8 * length):]
	test_bars = sum([len(open(f).readlines()) for f in testData])

	print("------")
	print(len(trainData), '(', train_bars, ')', len(testData), '(', test_bars, ')')
	print("-----")

	return trainData, testData


def get_num_splits(train_files, test_files):
	return [(num_bars(x), num_bars(y), num_bars(x)/(num_bars(x) + num_bars(y))) for x,y in zip(test_files,train_files)]


def kfold_split(directory, n):
	musicFiles = sorted(glob.glob(os.path.join(directory, "*")))
	total_bars = num_bars(musicFiles)

	test_bars = total_bars/n

	train_files = []
	test_files = []

	remaining_files = musicFiles
	for i in range(n):
		j = 0

		test_set = remaining_files[:j]
		while num_bars(test_set) < 0.95*test_bars and j <= len(remaining_files):
			j += 1
			test_set = remaining_files[:j]

		test_files.append(test_set)
		train_files.append(list(set(musicFiles) - set(test_set)))

		remaining_files = remaining_files[j:]

	return train_files, test_files


# def intra_song_split(directory):
# 	musicFiles = sorted(glob.glob(os.path.join(directory, "*")))
# 	total_bars = num_bars(musicFiles)
#


def fitModel(train, test, model=None, data_type=None, activation=None, kernel=None, ngramlength=4, num_notes=None,
			 padval=0, chords_in_ngram=False, notes=None, softmax=False, oversampling=False):
	if model == KO:
		if os.path.isfile(PICKLES + "knn.pkl"):
			os.remove(PICKLES + "knn.pkl")
		if os.path.isfile(PICKLES + "omm.pkl"):
			os.remove(PICKLES + "omm.pkl")

	bar_sequences_train, chord_sequences_train = parse_data(train, num_notes=num_notes, padval=padval)
	bar_sequences_test, chord_sequences_test = parse_data(test, num_notes=num_notes, padval=padval)

	obj = model(data_type=data_type, activation=activation, kernel=kernel, ngramlength=ngramlength,
				chords_in_ngram=chords_in_ngram, notes=notes, softmax=softmax, oversampling=oversampling)
	obj.fit(bar_sequences_train, chord_sequences_train)

	train_score = obj.score(bar_sequences_train, chord_sequences_train)
	test_score = obj.score(bar_sequences_test, chord_sequences_test)

	train_score_string = "({0:.3f}".format(train_score[0]) + ", " + "{0:.3f})".format(train_score[1]) \
		if isinstance(train_score, tuple) else "{0:.3f}".format(train_score)
	test_score_string = "({0:.3f}".format(test_score[0]) + ", " + "{0:.3f})".format(test_score[1]) \
		if isinstance(train_score, tuple) else "{0:.3f}".format(test_score)

	return {'train': train_score_string, 'test': test_score_string}


def get_model_info(model_dict, num_notes_val, ngramlength_val):
	model_name = model_dict.get('model', None)
	data_type = model_dict.get('data_type', None)
	activation = model_dict.get('activation', None)
	kernel = model_dict.get('kernel', None)
	num_notes = model_dict.get('num_notes', num_notes_val)
	padval = model_dict.get('padval', -1)
	ngramlength = model_dict.get('ngramlength', ngramlength_val)
	chords_in_ngram = model_dict.get('chords_in_ngram', False)
	softmax = model_dict.get('softmax', False)
	# oversampling = model_dict.get('oversampling', False)
	oversampling = False
	actval = activation if activation else kernel

	return [model_name, data_type, activation, kernel, num_notes, padval, ngramlength, chords_in_ngram,
			softmax, oversampling, actval]


def evaluate_models(train, test, num_notes_val=4, ngramlength_val=4):
	model_list = yaml.load(open(os.path.join(MODELS, "model_configs.yaml"), "r"))
	results = []

	for model_dict in model_list.get('models'):
		if model_dict.get('is_enabled', True):
			model_name, data_type, activation, kernel, num_notes, padval, \
			ngramlength, chords_in_ngram, softmax, \
			oversampling, actval = get_model_info(model_dict, num_notes_val, ngramlength_val)

			scores = fitModel(model=model_class[model_name], data_type=data_type, activation=activation,
							  kernel=kernel, train=train, test=test, num_notes=num_notes, padval=padval,
							  ngramlength=ngramlength, chords_in_ngram=chords_in_ngram, notes=num_notes,
							  softmax=softmax, oversampling=False)

			result = [model_name, data_type, num_notes, ngramlength, actval, chords_in_ngram, softmax, oversampling,
					  scores]

			results.append(result)

	sorted_results = sorted(results, key=lambda x: x[8]['test'])

	return sorted_results


def print_results(sorted_results):
	print("".join(word.ljust(20) for word in ['MODEL', 'DATA_TYPE', 'NOTES', 'NGRAMLENGTHVAL', 'ACTIVATION/KERNEL',
											  'CHORDS_IN_NRGAM', 'SOFTMAX', 'SMOTE', 'SCORES']))

	print('\n\nSORTED:')
	for result in sorted_results:
		print("".join(word.ljust(20) for word in [str(x) for x in result]))

	print('\n\nBEST :')
	print("".join(word.ljust(20) for word in [str(x) for x in sorted_results[-1]]))


def random_split_test(directory):
	train, test = randomSplit(directory=directory)
	sorted_results = evaluate_models(train=train, test=test, num_notes_val=4, ngramlength_val=5)
	return sorted_results


if __name__ == "__main__":
	sorted_results = random_split_test(directory=directories.PROCESSED_CHORDS_MULTI_OCTAVE)
	print_results(sorted_results)
