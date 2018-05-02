import glob
from musicai.main.constants.directories import *
import os.path
files = glob.glob(os.path.join(RAW,"*.csv"))

transposes = range(1,12)	# for every scale after C


for song in files:
	print(song)
	songfile = open(song)
	tempoline = songfile.readline()
	lines = songfile.readlines()
	songfile.close()
	notes = []
	for line in lines:
		note_info = line.strip().split(",")
		notes.append(note_info)
	
	for transpose in transposes:
		g = open(song.split(".csv")[0] + str(transpose)+".csv", "w")
		g.write(tempoline)
		for note in notes:
			g.write(str(int(note[0]) + transpose) + "," + note[1] + "," + note[2] + "\n")
		g.close()
