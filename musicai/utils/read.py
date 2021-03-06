import pygame
import pygame.midi
from pygame.locals import *
import glob

fname = input("Enter name of song : ").strip()
tempo = input("Enter the tempo : ").strip()

f = open(fname +  ".csv","w")
f.write("Tempo," + tempo  + "\n")


pygame.init()
pygame.midi.init()

# constants
SILENCE_THRESH = 5000
INPUT_ID = 3


inp = pygame.midi.Input( INPUT_ID )

last_event = 0
while True:
	if inp.poll():
		midi_event = inp.read(1)
		if midi_event[0][0][0] == 144:
			last_event = midi_event[0][1]
			f.write(str(midi_event[0][0][1]) + "," + str(midi_event[0][0][2]) + "," + str(midi_event[0][1]) + "\n")		# note,velocity,timestamp
		elif (midi_event[0][1] - last_event) > SILENCE_THRESH: 
			break

print("DONE")

f.close()
del inp
pygame.midi.quit()
pygame.quit()


