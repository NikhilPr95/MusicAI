from musicai.main.lib import knn
from musicai.main.models.base  import Base
from musicai.main.constants import directories
from sklearn.neighbors import KNeighborsClassifier
from musicai.main.lib.input_vectors import sequence_vectors
import glob
from pickle import load, dump
from musicai.main.constants.directories import *


class KNN(Base):

	def __init__(self):
		Base.__init__(self)
		self.clf = None

	def fit(self, X, y):
		if glob.glob(os.path.join(directories.PICKLES, 'knn.pkl')):
			self.clf = load(open(os.path.join(directories.PICKLES, 'knn.pkl'), "rb"))
		else:
			clf = KNeighborsClassifier(n_neighbors=3)
			clf.fit(X, y)
			self.clf = clf
			# print("TEST")
			dump(self.clf, open(os.path.join(directories.PICKLES, 'knn.pkl'), "wb"))

	def predict(self, x):
		outp = self.clf.predict([x])
		return outp