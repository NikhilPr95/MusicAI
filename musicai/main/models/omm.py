from musicai.main.lib import knn
from musicai.main.models.base  import Base
from musicai.main.constants import directories
from sklearn.neighbors import KNeighborsClassifier
from musicai.main.lib.input_vectors import sequence_vectors
import glob
from pickle import load, dump
from musicai.main.constants.directories import *
from musicai.main.lib.markov import transition_matrices

class OMM(Base):

    def __init__(self):
        Base.__init__(self)
        self.clf = None

    def fit(self, X, y=None):
        #X is chord sequences
        if glob.glob(os.path.join(directories.PICKLES, 'omm.pkl')):
            self.clf = load(open(os.path.join(directories.PICKLES, 'omm.pkl'), "rb"))
        else:
            data = transition_matrices(X)
            dump(data, open(os.path.join(directories.PICKLES, 'omm.pkl'), "wb"))
            self.clf = data

    def predict(self, x):
        max_val = 0
        key = "X"
        for each_chord in self.clf[1][x]:
            if max_val < self.clf[1][x][each_chord]:
                key = each_chord
                max_val = self.clf[1][x][each_chord]

        return key
