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


def splitData(dir):
	musicFiles_ = glob.glob(os.path.join(dir, "*"))
	random.shuffle(musicFiles_)
	musicFiles = [f for f in musicFiles_ if len(open(f).readlines()) > BAR_THRESHOLD]
	length = len(musicFiles)

	trainData = musicFiles[:int(0.8 * length)] + list(set(musicFiles_) - set(musicFiles))
	valData = []  # musicFiles[int(0.6 * length):int(0.8 * length)]
	testData = musicFiles[int(0.8 * length):]

	print("------")
	print(len(trainData), len(testData))
	print("-----")

	return trainData, valData, testData


def fitModel(train, test, model=None, data_type=None, activation=None, kernel=None, ngramlength=4, padding=None,
			 padval=0, chords_in_ngram=False):
	if model == KO:
		if os.path.isfile(PICKLES + "knn.pkl"):
			os.remove(PICKLES + "knn.pkl")
		if os.path.isfile(PICKLES + "omm.pkl"):
			os.remove(PICKLES + "omm.pkl")

	bar_sequences_train, chord_sequences_train = parse_data(train, padding=padding, padval=padval)
	bar_sequences_test, chord_sequences_test = parse_data(test, padding=padding, padval=padval)

	obj = model(data_type=data_type, activation=activation, kernel=kernel, ngramlength=ngramlength, chords_in_ngram=chords_in_ngram)
	obj.fit(bar_sequences_train, chord_sequences_train)

	train_score = obj.score(bar_sequences_train, chord_sequences_train)
	test_score = obj.score(bar_sequences_test, chord_sequences_test)

	train_score_string = "({0:.3f}".format(train_score[0]) + ", " + "{0:.3f})".format(train_score[1]) \
		if isinstance(train_score, tuple) else "{0:.3f}".format(train_score)
	test_score_string = "({0:.3f}".format(test_score[0]) + ", " + "{0:.3f})".format(test_score[1]) \
		if isinstance(train_score, tuple) else "{0:.3f}".format(test_score)

	return {'train': train_score_string, 'test': test_score_string}


model_class = {'KO': KO, 'SVM': SVM, 'MLP': MLP, 'LogReg': LogReg, 'PyHMM': PyHMM}

if __name__ == "__main__":
	train, val, test = splitData(directories.PROCESSED_CHORDS)

	print("".join(word.ljust(20) for word in ['MODEL', 'DATA_TYPE', 'ACTIVATION/KERNEL', 'CHORDS_IN_NRGAM', 'SCORES']))
	model_list = yaml.load(open(os.path.join(MODELS, "model_configs.yaml"), "r"))
	for model_dict in model_list.get('models'):
		if model_dict.get('is_enabled', True):
			model_name = model_dict.get('model', None)
			data_type = model_dict.get('data_type', None)
			activation = model_dict.get('activation', None)
			kernel = model_dict.get('kernel', None)
			padding_string = model_dict.get('padding', None)
			padding = int(padding_string) if padding_string else None
			padval = int(model_dict.get('padval', 0))
			ngramlength = int(model_dict.get('ngramlength', 4))
			chords_in_ngram = model_dict.get('chords_in_ngram', False)
			actval = activation if activation else kernel

			if model_name in ['LogReg']:
				scores = fitModel(model=model_class[model_name], data_type=data_type, activation=activation, kernel=kernel,
								  train=train, test=test, padding=padding, padval=padval,
								  ngramlength=ngramlength, chords_in_ngram=chords_in_ngram)

				print("".join(word.ljust(20) for word in [model_name, str(data_type), str(actval), str(chords_in_ngram), str(scores)]))
