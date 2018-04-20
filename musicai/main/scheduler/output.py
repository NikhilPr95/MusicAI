# renders left hand arpeggio to keyboard
# contains pygame code
# for now just print chord labels
# read from shared memory

import pygame.midi
import time, random
import SharedArray as sa
from musicai.main.gui.main import App, buttons

def sendOutput():
    # puts the output to the keyboard, gives the midi output back after reading from shared memory
    # poll shared mem
    # notesToPlay = [[48,100],[48,100],[48,100],[48,100]]
    while (chordsToPlay[0] == 0.0):
        continue
    print(chordsToPlay)
    while (not (chordsToPlay[0] == -1.0)):
        for i in range(len(chordsToPlay)):
            print(chordsToPlay[i])
            #out.note_on(48 + int(chordsToPlay[i]), velocity=60)
            buttons[chordsToPlay[i]].invoke()
            App.update()
            time.sleep((60 / tempo))
            print("OUT")
            #out.note_off(48 + int(chordsToPlay[i]), velocity=0)
	
if __name__ == "__main__":
    tempo = 80 #define
    #pygame.midi.init()
    print("test")
    App.update()
    #chordsToPlay = sa.attach("shm://notes")
    chordsToPlay = [1,1,1,3,3,3,3,3,3,3,3]
    #out = pygame.midi.Output(2)
    sendOutput()
    #del out
    #pygame.midi.quit()
