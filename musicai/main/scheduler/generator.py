# renders left hand arpeggio to keyboard
# contains pygame code
# for now just print chord labels
# read from shared memory

import pygame.midi
import time, random
import SharedArray as sa
from musicai.main.generation.client import *
from musicai.utils.chords import get_notes, get_chord_mapping
from musicai.main.lib.input_vectors import reduce


def sendOutput():
	# puts the output to the keyboard, gives the midi output back after reading from shared memory
	# poll shared mem
	# notesToPlay = [[48,100],[48,100],[48,100],[48,100]]
	while (not (leftHandNotes[0] == -1.0)):
		for i in range(len(leftHandNotes)):
			out.note_on(60 + int(rightHandNotes[i][0][0]), velocity=rightHandNotes[i][1])
			out.note_on(48 + int(rightHandNotes[i]), velocity=rightHandNotes[i][1])
			# time.sleep((60 / tempo))
			time.sleep(rightHandNotes[i][0][2])
			print("OUT")
			out.note_off(60 + int(rightHandNotes[i][0][0]), velocity=0)
			out.note_off(48 + int(leftHandNotes[i]), velocity=0)

if __name__ == "__main__":
	tempo = 80  # define
	pygame.midi.init()
	#chordsToPlay = sa.attach("shm://notes")
	chordMaps = get_chord_mapping()
	ngramsGenerated = gen_trigram()
	rightHandNotes = [i[0] for i in ngramsGenerated]  # generating notes only.
	print("Right hand notes ",rightHandNotes)
	leftHandNotes = [(get_notes(ngram[1],chordMaps)) for ngram in ngramsGenerated]
	leftHandNotes = flatten(leftHandNotes)
	print("Left hand notes : ", leftHandNotes)
	out = pygame.midi.Output(2)
	sendOutput()
	del out
	pygame.midi.quit()
