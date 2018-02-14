from sklearn.neighbors import KNeighborsClassifier
from lib.input_vectors import sequence_vectors
import glob
from pickle import load, dump

def knn_predict(right_hand_notes):
	if glob.glob("knn.pkl") :
		clf = load(open("knn.pkl","rb"))
	else:
		data = sequence_vectors("../data/processed_chords/")
		X = data[0]
		y = data[1]
		
		clf = KNeighborsClassifier(n_neighbors=3)
		clf.fit(X,y)
		dump(clf, open("knn.pkl","wb"))
	return clf.predict([right_hand_notes])


