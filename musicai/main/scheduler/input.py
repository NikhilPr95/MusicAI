# reads input from the keyboard
# pygame code comes here
import pygame
import pygame.midi
from pygame.locals import *
import signal

from musicai.main.lib.predict import predict


tempo = int(input("Enter the tempo : ").strip())
bar_length = int((4/tempo) * 60)  #bar length in seconds
print(bar_length)
def push_notes(signum, frame):
	# read midi data of global array
	# call predict
	# re-initialize array for next bar
	# set another alarm
	global notes_of_bar
	#print("NOTES OF BAR : ", notes_of_bar)
	print(predict(notes_of_bar))
	notes_of_bar = []
	signal.alarm(bar_length)

notes_of_bar = []	

signal.signal(signal.SIGALRM, push_notes)
signal.alarm(bar_length)

pygame.init()
pygame.midi.init()

INPUT_ID = 3
SILENCE_THRESH = 5000
inp = pygame.midi.Input( INPUT_ID )

last_event = 0
while True:
	if inp.poll():
		midi_event = inp.read(1)
		if midi_event[0][0][0] == 144:
			last_event = midi_event[0][1]
			notes_of_bar.append(midi_event[0][0][1])
		elif (midi_event[0][1] - last_event) > SILENCE_THRESH:
			signal.alarm(0) 
			break

print("DONE")

del inp
pygame.midi.quit()
pygame.quit()

