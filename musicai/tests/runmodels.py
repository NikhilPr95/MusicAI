import csv
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


def evaluate_models(train_list, test_list, data_list={'sequence', 'ngram', 'ngram_notes', 'current_bar'},
                    num_notes_val=4, ngramlength_val=1, logfile=None):
    logcsv = open(logfile.name[:-3] + '.csv', 'w+', newline='')
    csvwriter = csv.writer(logcsv)

    header_list = ['MODEL', 'DATA_TYPE', 'NOTES', 'NGRAMLENGTHVAL', 'ACTIVATION/KERNEL',
                                    'CHORDS_IN_NRGAM', 'SOFTMAX', 'SCORES', 'AVG. SCORE']
    header_string = "".join(
        word.ljust(20) for word in header_list)
    print(header_string)
    print('logfile:', logfile, logfile is not None)
    if logfile is not None:
        logfile.write(header_string)
        logfile.write('\n')
        csvwriter.writerow(header_list)

    model_list = yaml.load(open(os.path.join(MODELS, "model_configs.yaml"), "r"))
    results = []

    for model_dict in model_list.get('models'):
        if model_dict.get('is_enabled', True):
            model_name, data_type, activation, kernel, num_notes, padval, \
            ngramlength, chords_in_ngram, softmax, \
            oversampling, actval = get_model_info(model_dict, num_notes_val, ngramlength_val)

            if data_type in data_list:

                scores = []
                for train, test in zip(train_list, test_list):
                    score = fitModel(model=model_class[model_name], data_type=data_type, activation=activation,
                                     kernel=kernel, train=train, test=test, num_notes=num_notes, padval=padval,
                                     ngramlength=ngramlength, chords_in_ngram=chords_in_ngram, notes=num_notes,
                                     softmax=softmax, oversampling=False)
                    scores.append(score)

                avg_score = {'train': "{0:.4f}".format(sum([float(score['train']) for score in scores]) / len(scores)),
                             'test': "{0:.4f}".format(sum([float(score['test']) for score in scores]) / len(scores))}

                result = [model_name, data_type, num_notes, ngramlength, actval, chords_in_ngram, softmax,
                          [(s['train'], s['test']) for s in scores], (avg_score['train'], avg_score['test'])]

                res_string = "".join(word.ljust(20) for word in [str(x) for x in result])
                print(res_string)
                if logfile is not None:
                    logfile.write(res_string)
                    logfile.write('\n')
                    csvwriter.writerow([str(x) for x in result])

                results.append(result)

    sorted_results = sorted(results, key=lambda x: x[8][1])

    return sorted_results


def print_results(sorted_results, logfile=None):
    logcsv = open(logfile.name[:-3] + '_sorted.csv', 'w+', newline='')
    csvwriter = csv.writer(logcsv)

    header_list = ['MODEL', 'DATA_TYPE', 'NOTES', 'NGRAMLENGTHVAL', 'ACTIVATION/KERNEL',
                                    'CHORDS_IN_NRGAM', 'SOFTMAX', 'SCORES', 'AVG. SCORE']
    header_string = "".join(
        word.ljust(20) for word in header_list)
    print(header_string)
    logfile.write(header_string)
    logfile.write('\n')
    csvwriter.writerow(header_list)

    print('\n\nSORTED:')
    logfile.write('\n\nSORTED:\n')

    for result in sorted_results:
        res_string = "".join(word.ljust(20) for word in [str(x) for x in result])
        print(res_string)
        logfile.write(res_string)
        logfile.write('\n')
        csvwriter.writerow([str(x) for x in result])

    print('\n\nBEST:\n')
    logfile.write('\n\nBEST:\n')
    print("".join(word.ljust(20) for word in [str(x) for x in sorted_results[-1]]))
    logfile.write("".join(word.ljust(20) for word in [str(x) for x in sorted_results[-1]]))
    logfile.write('\n')


def random_split_test(train, test):
    sorted_results = evaluate_models(train_list=[train], test_list=[test], num_notes_val=4, ngramlength_val=5)
    return sorted_results


def kfold_split_test(directory, n, data_list=None, logfile=None):
    train_file_sets, test_file_sets, ratios = kfold_split(directory, n)
    logfile.write('RATIOS-\n')
    logfile.write(repr(ratios))
    logfile.write('\n\n')

    if data_list == None:
        sorted_results = evaluate_models(train_file_sets, test_file_sets, logfile=logfile)
    else:
        sorted_results = evaluate_models(train_file_sets, test_file_sets, data_list=data_list, logfile=logfile)
    return sorted_results


def get_all_results():
    for dir in [directories.PROCESSED_CHORDS,
                directories.PROCESSED_CHORDS_RHYMES,
                directories.PROCESSED_CHORDS_POP,
                # directories.PROCESSED_CHORDS_IMPROV,
                directories.PROCESSED_CHORDS_MULTI_OCTAVE,
                directories.PROCESSED_CHORDS_MULTI_OCTAVE_RHYMES,
                directories.PROCESSED_CHORDS_MULTI_OCTAVE_POP,
                directories.PROCESSED_CHORDS_MULTI_OCTAVE_IMPROV]:
        print('DIR:', dir)
        for data_type in ['current_bar', 'sequence', 'ngram', 'ngram_notes']:
            print('DATA_TYPE:', data_type)
            for k in [5, 10]:
                print('K:', k)
                if data_type == 'ngram_notes':
                    pass
                else:
                    logdir = os.path.basename(dir[:-1])
                    logfile = open(os.path.join(directories.RESULTS, logdir, data_type + str(k) + '.txt'), 'w+')
                    sorted_results = kfold_split_test(dir, k, {data_type}, logfile)
                    print_results(sorted_results, logfile)

    for train, test in [
        (directories.PROCESSED_CHORDS_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_RHYMES_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_RHYMES_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_POP_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_POP_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_IMPROV_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_IMPROV_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_MULTI_OCTAVE_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_MULTI_OCTAVE_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_MULTI_OCTAVE_RHYMES_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_MULTI_OCTAVE_RHYMES_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_MULTI_OCTAVE_POP_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_MULTI_OCTAVE_POP_SONG_SPLIT_TEST),
        (directories.PROCESSED_CHORDS_MULTI_OCTAVE_IMPROV_SONG_SPLIT_TRAIN,
         directories.PROCESSED_CHORDS_MULTI_OCTAVE_IMPROV_SONG_SPLIT_TEST)
    ]:
        print('DIR:', test)
        for data_type in ['sequence', 'ngram', 'current_bar', 'ngram_notes']:
            print('DATA_TYPE:', data_type)
            if data_type == 'ngram_notes':
                pass
            else:
                logdir = os.path.basename(test[:-1])
                logfile = open(os.path.join(directories.RESULTS, logdir, data_type + '.txt'), 'w+')
                sorted_results = evaluate_models(train, test, data_list={data_type}, logfile=logfile)
                print_results(sorted_results, logfile)


if __name__ == "__main__":
    get_all_results()
