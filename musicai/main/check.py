from musicai.main.constants import directories
from musicai.main.lib.input_vectors import sequence_vectors

file = directories.PROC_CHORDS + '/demons.csv.formatted'

sequence_vectors(file)
