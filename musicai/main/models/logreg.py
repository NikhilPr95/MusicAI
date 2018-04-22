import numpy as np
from musicai.main.constants.values import SIMPLE_CHORDS
from musicai.main.lib.input_vectors import ngram_vector, create_ngram_feature_matrix, \
    create_standard_feature_matrix
from musicai.main.models.base import Base
from musicai.utils.general import flatten, get_softmax
from sklearn.linear_model import LogisticRegression


class LogReg(Base):
    def __init__(self, ngramlength=4, activation='relu', data_type='current_bar', kernel=None, chords_in_ngram=False,
                 notes=None, softmax=False, oversampling=False):
        Base.__init__(self)
        self.clf = None
        self.activation = activation
        self.ngramlength = ngramlength
        self.data_type = data_type
        self.kernel = kernel
        self.chords_in_ngram = chords_in_ngram
        self.notes = notes
        self.softmax = softmax
        self.oversampling = oversampling

    def fit(self, bar_sequences, chord_sequences):
        if self.kernel is not None:
            raise Exception("Model does not support {} kernel".format(self.kernel))
        if self.data_type == 'ngram_notes':
            X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, ngramlength=self.ngramlength,
                                               chords_in_ngram=self.chords_in_ngram, notes=self.notes, oversampling=self.oversampling)
        elif self.data_type == 'current_bar':
            if self.chords_in_ngram is not False:
                raise Exception("Model does not support chords in ngram with current bar")
            X, y = create_standard_feature_matrix(bar_sequences, chord_sequences, num_notes=self.notes, oversampling=self.oversampling)
        else:
            raise Exception("Model does not support {} data type".format(self.data_type))

        if self.softmax:
            X = get_softmax(X)

        X = np.array(X)
        y = np.array(y)

        self.clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
        # self.clf = LogisticRegression(max_iter=1000)
        self.clf.fit(X, y)

    # print("score:", self.clf.score(X, y))

    def predict(self, input):
        chord = self.clf.predict([input])
        return SIMPLE_CHORDS[chord[0]]

    def score(self, bar_sequences, chord_sequences):
        if self.data_type == 'current_bar':
            X, y = create_standard_feature_matrix(bar_sequences, chord_sequences, num_notes=self.notes)
        elif self.data_type == 'ngram_notes':
            X, y = create_ngram_feature_matrix(bar_sequences, chord_sequences, ngramlength=self.ngramlength,
                                               chords_in_ngram=self.chords_in_ngram, notes=self.notes)
        else:
            raise Exception("Model does not support {} data type".format(self.data_type))

        return self.clf.score(X, y)
