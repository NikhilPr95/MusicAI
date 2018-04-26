import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM as LSTM_keras
from musicai.main.models.base import Base
from keras.preprocessing import sequence
import numpy as np
import os,glob
from musicai.main.constants import directories
from musicai.main.constants.directories import *
from pickle import load, dump

class LSTM(Base):
	def __init__(self):
		Base.__init__(self)
		self.model = Sequential()
		self.model.add(LSTM_keras(50,input_shape=(4, 1), return_sequences=False))
		self.model.add(Dense(7, activation = "softmax"))
		self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
	def fit(self, x, y ):
		if glob.glob(os.path.join(directories.PICKLES, 'lstm.pkl')):
			self.model = load(open(os.path.join(directories.PICKLES, 'lstm.pkl'), "rb"))
		else:
			x = np.array(x)
			x = x.reshape(x.shape[0], x.shape[1], 1)
			print("After reshape X:",x)
			self.model.fit(x, y, nb_epoch=100, batch_size=64)
			scores = self.model.evaluate(x, y, verbose=0)
			print("Accuracy: %.2f%%" % (scores[1]*100))
			dump(self.model, open(os.path.join(directories.PICKLES, 'lstm.pkl'), "wb"))
		
	def predict(self, y):
		print(y)
		y = np.array(y)
		y = y.reshape(1, y.shape[1], 1)
		print(y.shape)
		pred = self.model.predict(np.array(y))
		print(pred.argmax())
		print(pred)
		return pred.argmax()
'''
numpy.random.seed(7)
(X_train, y_train), (X_test, y_test) = imdb.load_data(path=PATH, num_words=top_words)
#print(len(X_train), len(y_train), len(X_test), len(y_test))
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_length)
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(50))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
'''