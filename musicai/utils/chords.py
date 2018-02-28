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
	print(notes)
	notes = [i for i in range(len(notes)) if notes[i] == "1"] # starting with c
	#apply broken chords logic

	random.shuffle(notes)
	
	return notes

