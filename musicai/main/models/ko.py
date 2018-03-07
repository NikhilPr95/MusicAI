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

    def fit(self, bar_sequences, chord_sequences):
        self.knn.fit(bar_sequences, chord_sequences)
        self.omm.fit(chord_sequences)

    def predict(self, bar_sequence):
        if len(bar_sequence) < MAX_NOTES:
            bar_sequence += [0] * (MAX_NOTES - len(bar_sequence))  # pad with zeros
        if len(bar_sequence) > MAX_NOTES:
            bar_sequence = bar_sequence[:MAX_NOTES]  # crop perhaps
        # push to shared memory instead of returning here
        knn_result = self.knn.predict(bar_sequence)[0]
        return knn_result,self.omm.predict(knn_result)