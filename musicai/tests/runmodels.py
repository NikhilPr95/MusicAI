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
from musicai.utils.files import randomSplit, num_bars, kfold_split

model_class = {'KO': KO, 'SVM': SVM, 'MLP': MLP, 'LogReg': LogReg, 'PyHMM': PyHMM}


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
	oversampling = model_dict.get('oversampling', False)
	# oversampling = False
	actval = activation if activation else kernel

	return [model_name, data_type, activation, kernel, num_notes, padval, ngramlength, chords_in_ngram,
	        softmax, oversampling, actval]


def evaluate_models(train_list, test_list, num_notes_val=4, ngramlength_val=4):
	print("".join(word.ljust(20) for word in ['MODEL', 'DATA_TYPE', 'NOTES', 'NGRAMLENGTHVAL', 'ACTIVATION/KERNEL',
	                                          'CHORDS_IN_NRGAM', 'SOFTMAX', 'SCORES']))
	model_list = yaml.load(open(os.path.join(MODELS, "model_configs.yaml"), "r"))
	results = []

	for model_dict in model_list.get('models'):
		if model_dict.get('is_enabled', True):
			model_name, data_type, activation, kernel, num_notes, padval, \
			ngramlength, chords_in_ngram, softmax, \
			oversampling, actval = get_model_info(model_dict, num_notes_val, ngramlength_val)

			if data_type == 'current_bar':
				scores = []
				for train, test in zip(train_list, test_list):
					score = fitModel(model=model_class[model_name], data_type=data_type, activation=activation,
					                 kernel=kernel, train=train, test=test, num_notes=num_notes, padval=padval,
					                 ngramlength=ngramlength, chords_in_ngram=chords_in_ngram, notes=num_notes,
					                 softmax=softmax, oversampling=False)
					scores.append(score)

				avg_score = {'train': sum([score['train'] for score in scores]) / len(scores),
				             'test': sum([score['test'] for score in scores]) / len(scores)}

				result = [model_name, data_type, num_notes, ngramlength, actval, chords_in_ngram, softmax,
				          scores, avg_score]

				print("".join(word.ljust(20) for word in [str(x) for x in result]))
				results.append(result)

	sorted_results = sorted(results, key=lambda x: x[8]['test'])

	return sorted_results


def print_results(sorted_results):
	print("".join(word.ljust(20) for word in ['MODEL', 'DATA_TYPE', 'NOTES', 'NGRAMLENGTHVAL', 'ACTIVATION/KERNEL',
	                                          'CHORDS_IN_NRGAM', 'SOFTMAX', 'SCORES']))

	print('\n\nSORTED:')
	for result in sorted_results:
		print("".join(word.ljust(20) for word in [str(x) for x in result]))

	print('\n\nBEST :')
	print("".join(word.ljust(20) for word in [str(x) for x in sorted_results[-1]]))


def random_split_test(train, test):
	sorted_results = evaluate_models(train_list=[train], test_list=[test], num_notes_val=4, ngramlength_val=5)
	return sorted_results


def kfold_split_test(directory, n):
	train_file_sets, test_file_sets = kfold_split(directory, n)
	sorted_results = evaluate_models(train_file_sets, test_file_sets)
	return sorted_results


if __name__ == "__main__":
	# train, test = randomSplit(directory=directories.PROCESSED_CHORDS_MULTI_OCTAVE)
	# train = glob.glob(os.path.join(directories.PROCESSED_CHORDS_MULTI_OCTAVE_SONG_SPLIT_TRAIN, "*"))
	# test = glob.glob(os.path.join(directories.PROCESSED_CHORDS_MULTI_OCTAVE_SONG_SPLIT_TEST, "*"))
	# print('train ', len(train), num_bars(train), 'test ', len(test), num_bars(test))

	# sorted_results = random_split_test(train, test)
	# print_results(sorted_results)

	sorted_results = kfold_split_test(directories.PROCESSED_CHORDS_MULTI_OCTAVE, 5)
	print_results(sorted_results)
