# renders left hand arpeggio to keyboard
# contains pygame code
# for now just print chord labels
# read from shared memory

import pygame.midi
import time, random
import SharedArray as sa

def sendOutput():
    # puts the output to the keyboard, gives the midi output back after reading from shared memory
    # poll shared mem
    # notesToPlay = [[48,100],[48,100],[48,100],[48,100]]
    global buttons, App
    while (chordsToPlay[0] == 0.0):
        continue
    print(chordsToPlay)
    while (not (chordsToPlay[0] == -1.0)):
        for i in range(len(chordsToPlay)):
            print(chordsToPlay[i])
            out.note_on(48 + int(chordsToPlay[i]), velocity=random.randint(45,75))
            buttons[int(chordsToPlay[i])].invoke()
            App.update()
            time.sleep((60 / tempo) * 0.5)
            print("OUT")
            out.note_off(48 + int(chordsToPlay[i]), velocity=0)

        for i in range(len(chordsToPlay)):
            print(chordsToPlay[i])
            random.shuffle(chordsToPlay)
            out.note_on(48 + int(chordsToPlay[i]), velocity=random.randint(45,75))
            buttons[int(chordsToPlay[i])].invoke()
            App.update()
            time.sleep((60 / tempo) * 0.5)
            print("OUT")
            out.note_off(48 + int(chordsToPlay[i]), velocity=0)

	
def output(App_parameter,buttons_parameter, tempo_temp):
    global chordsToPlay , out, tempo, App, buttons
    #tempoQueue = sa.attach("shm://tempo")
    #while(tempoQueue[0] == 0.0):
    #    continue
    #tempo = tempoQueue[0] #define
    tempo = tempo_temp
    App = App_parameter
    buttons = buttons_parameter
    print(tempo)
    pygame.midi.init()
    print("test")
    time.sleep(1)
    chordsToPlay = sa.attach("shm://notes")
    out = pygame.midi.Output(2)
    sendOutput()
    del out
    pygame.midi.quit()
