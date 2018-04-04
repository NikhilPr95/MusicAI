import glob

import os
from musicai.main.constants import directories
# from musicai.main.lib.input_vectors import parse_data
from musicai.utils.general import flatten


def get_chord_mapping():
	chords_dataset = open("musicai/data/chords/chords.data", "r").readlines()[::-1]
	chords = {}
	for chord_temp in chords_dataset:
		chord_arr = chord_temp.split(",")
		chords[chord_arr[-1].strip()] = "".join(list(map(lambda x: "1" if x == "YES" else "0", chord_arr[2:14])))
	return chords	


def get_notes(chord, chords):
	import random
	notes = chords[chord]
	# print(notes)
	notes = [i for i in range(len(notes)) if notes[i] == "1"] # starting with c
	#apply broken chords logic

	random.shuffle(notes)
	
	return notes


def reduce(chord_string):
	return chord_string[0]


def count_chords():
	musicfiles = glob.glob(os.path.join(directories.PROCESSED_CHORDS, "*"))
	# _, chord_sequences = parse_data(musicfiles)
	# chords = flatten(chord_sequences)
	# print(chords)

	chordDict = dict()
	for file_name in musicfiles:
		f = open(file_name)
		for line in f:
			chord = line.strip().split(",")[-1]
			if chord not in chordDict:
				chordDict[chord] = 0
			chordDict[chord] += 1
		f.close()

	return chordDict