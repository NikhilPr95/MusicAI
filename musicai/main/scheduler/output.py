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
    while (chordsToPlay[0] == 0.0):
        continue
    print(chordsToPlay)
    while (not (chordsToPlay[0] == -1.0)):
        chord = chordsToPlay[0]
        notes = getNotes(chord)
        for i in range(4):
            print(notes[i])
            out.note_on(int(notes[i]), velocity=60)
            time.sleep((60 / tempo))
            print("OUT")
            out.note_off(int(notes[i]), velocity=0)

def getNotes(chord):
    chords_dataset = open("../../data/chords/chords.data", "r")
    chords = {}
    for chord_temp in chords_dataset:
        chord_arr = chord_temp.split(",")
        chords[chord_arr[-1].strip()] = "".join(list(map(lambda x: "1" if x == "YES" else "0", chord_arr[2:14])))

    notes = chords[chord]
    notes = [i for i in range(len(chords)) if chords[i] == "1"] # starting with c
    #apply broken chords logic

    notes[:] = random.shuffle(notes)
    return notes


if __name__ == "__main__":
    tempo = 175.0 #define
    pygame.midi.init()
    chordsToPlay = sa.attach("shm://notes")
    for i in range(0, len(chordsToPlay)):
        chordsToPlay[i] = 0.0
    out = pygame.midi.Output(2)
    sendOutput()
    del out
    pygame.midi.quit()
