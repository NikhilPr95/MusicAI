from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors

file = directories.PROC_CHORDS# + '/demons.csv.formatted'

data, labels = sequence_vectors(file)

for row in zip(data, labels):
    print(row)
