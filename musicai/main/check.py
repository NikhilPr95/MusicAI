from main.lib.omm import omm_train
from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors

file = directories.PROC_CHORDS# + '/demons.csv.formatted'

data, labels = sequence_vectors(file)
# data, labels = sequence_vectors("../data/processed_chords/happy_birthday.csv.formatted")

for row in zip(data, labels):
    print(row)

omm_train()