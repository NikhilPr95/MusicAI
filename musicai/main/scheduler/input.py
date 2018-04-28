# reads input from the keyboard
# pygame code comes here
import pygame
import pygame.midi
from pygame.locals import *
import signal
import SharedArray as sa
import numpy as np
import sys
from musicai.main.lib.predict import predict
from musicai.utils.chords import get_notes, get_chord_mapping

'''
if len(sys.argv) > 1 and sys.argv[1] == "--test":
	print(predict([1, 2, 3, 4, 5, 6, 7]))
	sys.exit()
'''
def sendInput(App, buttons, tempo_temp):
	
	queue = []
	if bytes('notes','utf-8') not in [list(a)[0] for a in sa.list()]:
		queue = sa.create("shm://notes", 4)
	for i in range(0, len(queue)):
		queue[i] = 0.0
	
	#tempoQueue = sa.attach("shm://tempo")
	#while(tempoQueue[0] == 0.0):
	#	continue
	#tempo = tempoQueue[0]#int(input("Enter the tempo : ").strip())
	tempo = tempo_temp
	print(tempo)
	bar_length = int((4/tempo) * 60)  #bar length in seconds
	print(bar_length)
	
	chords = get_chord_mapping()
	def push_notes(signum, frame):
		# read midi data of global array
		# call predict
		# re-initialize array for next bar
		# set another alarm
		global notes_of_bar, chords
		#print("NOTES OF BAR : ", notes_of_bar)
		prediction = predict(notes_of_bar)
		
		print(prediction)
		notes = get_notes(prediction, chords)
		print(notes)
		
		#update the GUI
		#right hand notes
		[buttons[36 + (note % 12)].invoke() for note in notes_of_bar]
		
		#left hand notes
		[buttons[note % 12].invoke() for note in notes]
		
		App.update()
		
		
		for i in range(len(notes)):
			queue[i] = notes[i]
	
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
	App.update()
	
	last_event = 0
	while True:
		if inp.poll():
			midi_event = inp.read(1)
			if midi_event[0][0][0] == 144:
				last_event = midi_event[0][1]
				buttons[24+int(midi_event[0][0][1])].invoke()
				App.update()
				notes_of_bar.append(midi_event[0][0][1])
			elif (midi_event[0][1] - last_event) > SILENCE_THRESH:
				signal.alarm(0)
				queue[0] = -1.0
				break
	
	print("DONE")
	
	del inp
	pygame.midi.quit()
	pygame.quit()



