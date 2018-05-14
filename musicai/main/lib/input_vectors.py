import csv
import glob

import os

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from musicai.main.constants import directories
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.utils.chords import reduce
from musicai.utils.general import flatten


def sequence_vectors(csvfilepath, num_notes=None, chords=False, octave=False, reduce_chords=False,
                     padval=-1):  # num_notes is the len of the vector required
    def getdata(csvfile, data, labels, maxlen):
        rows = csv.reader(open(csvfile, "r"))

        for row in rows:
            right_note_inputs = row[0].split('-')
            if right_note_inputs[0] != '':
                bar = [int(note_val.split('|')[0]) for note_val in right_note_inputs if note_val.split('|')[1] != '0']
                if octave:
                    bar = [b % 12 for b in bar]

                if len(bar):
                    if len(bar) > maxlen:
                        maxlen = len(bar)

                    data.append(bar)
                    label = row[2]
                    if reduce_chords:
                        labels.append(reduce(label))
                    else:
                        labels.append(label)

        return maxlen

    data = []
    labels = []
    maxlen = 0

    if os.path.isfile(csvfilepath):
        maxlen = getdata(csvfilepath, data, labels, maxlen)

    elif os.path.isdir(csvfilepath):
        for csvfile in os.listdir(csvfilepath):
            if csvfile.endswith('csv.formatted'):
                maxlen = getdata(csvfilepath + '/' + csvfile, data, labels, maxlen)
    if num_notes:
        for bar in data:
            if len(bar) < num_notes:
                if chords:
                    bar.extend([62] * (num_notes - len(bar)))
                else:
                    bar.extend([padval] * (num_notes - len(bar)))
            else:
                del bar[num_notes:]
    return data, labels


def find_ngrams(input_list, n):
    return list([list(x) for x in zip(*[input_list[i:] for i in range(n)])])


def ngram_vector(sequences, n):
    sequence_ngrams = []
    for sequence in sequences:
        sequence_ngrams.append([])
        sequence_ngrams[-1] = [flatten(seq_ngram) for seq_ngram in find_ngrams(sequence, n)]

    return sequence_ngrams


def create_ngram_feature_matrix(bar_sequences, chord_sequences, ngramlength, chords_in_ngram=False, notes=1,
                                oversampling=False, previous=False):
    bar_sequence_ngrams, \
    chord_sequence_ngrams = ngram_vector(bar_sequences, ngramlength), ngram_vector(
        chord_sequences, ngramlength)

    all_bar_ngrams = flatten(bar_sequence_ngrams)
    all_chord_ngrams = flatten(chord_sequence_ngrams)

    X, y = [], []
    if previous == True:
        for bar_ngram, chord_ngram in zip(all_bar_ngrams[:-1], all_chord_ngrams[1:]):
            chord_ngram_numbers = [SIMPLE_CHORDS.index(c) for c in chord_ngram]
            if chords_in_ngram:
                X.append(bar_ngram + chord_ngram_numbers)
            else:
                X.append(bar_ngram)
            y.append(chord_ngram_numbers[-1])

    else:
        for bar_ngram, chord_ngram in zip(all_bar_ngrams, all_chord_ngrams):
            chord_ngram_numbers = [SIMPLE_CHORDS.index(c) for c in chord_ngram]
            if chords_in_ngram:
                X.append(bar_ngram + chord_ngram_numbers[:-1])
            else:
                X.append(bar_ngram)
            y.append(chord_ngram_numbers[-1])

    if oversampling:
        if oversampling == 'smote':
            sampler = SMOTE(random_state=42)
        elif oversampling == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif oversampling == 'random':
            sampler = RandomOverSampler(random_state=42)
        else:
            raise Exception('no oversampler {} '.format(oversampling))
        X_normalized, y_normalized = sampler.fit_sample(X, y)

        return X_normalized, y_normalized

    return X, y


def create_standard_feature_matrix(bar_sequences, chord_sequences, exclude=0, chord_label_offset=0, num_notes=-1,
                                   oversampling=False):
    X, y = [], []
    for bar_sequence, chord_sequence in zip(bar_sequences, chord_sequences):
        for i in range(len(chord_sequence) - exclude):
            X.append(bar_sequence[i][:num_notes]) if num_notes else X.append(bar_sequence[i])
            y.append(chord_sequence[i + chord_label_offset])

    if oversampling:
        if oversampling == 'smote':
            sampler = SMOTE(random_state=42)
        elif oversampling == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif oversampling == 'random':
            sampler = RandomOverSampler(random_state=42)
        else:
            raise Exception('no oversampler {} '.format(oversampling))
        X_normalized, y_normalized = sampler.fit_sample(X, y)

        return X_normalized, y_normalized

    return X, y


def get_sequences(bar_sequences, notes=1):
    return [flatten([bar[0:notes] for bar in bar_sequence]) for bar_sequence in bar_sequences]


def parse_data(csvfilepaths, octave=True, reduce_chords=True, chords=False, num_notes=None, padval=0):
    """
    Parses csvs and returns bar and chord seqeunces
    Args:
        csvfilepath: Path to music data csv

    Returns:
    Bar and chord sequences
    """
    bar_sequences = []
    chord_sequences = []
    for csvfile in csvfilepaths:
        data = sequence_vectors(csvfile, num_notes, chords, octave, reduce_chords, padval)
        bar_sequences.append(data[0])
        chord_sequences.append(data[1])

    return bar_sequences, chord_sequences
