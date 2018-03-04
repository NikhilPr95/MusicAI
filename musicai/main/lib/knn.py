from sklearn.neighbors import KNeighborsClassifier
from musicai.main.lib.input_vectors import sequence_vectors
import glob
from pickle import load, dump
from musicai.main.constants.directories import *

def knn_predict(right_hand_notes):
	if glob.glob("musicai/main/pickles/knn.pkl") :
		clf = load(open("musicai/main/pickles/knn.pkl","rb"))
	else:
		data = sequence_vectors("musicai/data/processed_chords", padding = 15)
		X = data[0]
		y = data[1]
		
		clf = KNeighborsClassifier(n_neighbors=3)
		clf.fit(X,y)
		
		#print("TEST")
		dump(clf, open("musicai/main/pickles/knn.pkl","wb"))
	outp = clf.predict([right_hand_notes])
	print("KNN : ", outp)
	return outp


