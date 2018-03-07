from musicai.main.models.base import Base
from musicai.main.models.omm import OMM
from musicai.main.models.knn import KNN
from musicai.main.lib.input_vectors import sequence_vectors, parse_data
from musicai.main.constants.values import MAX_NOTES

class KO(Base):
    def __init__(self):
        Base.__init__(self)
        self.knn = KNN()
        self.omm = OMM()

    def fit(self, train , y=None):
        data = parse_data(train, padding=15)
        X = data[0]
        y = data[1]
        self.knn.fit(X, y)
        print(X)
        # omm
        chord_sequences = []
        for file_name in train:
            data = sequence_vectors(file_name)
            chord_sequences.append(data[1])

        self.omm.fit(chord_sequences)

    def predict(self, bar_sequence):
        if len(bar_sequence) < MAX_NOTES:
            bar_sequence += [0] * (MAX_NOTES - len(bar_sequence))  # pad with zeros
        if len(bar_sequence) > MAX_NOTES:
            bar_sequence = bar_sequence[:MAX_NOTES]  # crop perhaps
        # push to shared memory instead of returning here
        knn_result = self.knn.predict(bar_sequence)[0]
        return knn_result,self.omm.predict(knn_result)