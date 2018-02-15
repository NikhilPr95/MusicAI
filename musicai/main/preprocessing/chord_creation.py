import sys, difflib

g = open("../../data/processed/" + sys.argv[1].split("/")[-1], "r")
new_file = open("../../data/processed_chords/"+sys.argv[1].split("/")[-1], "w+")
chords_dataset = open("../../data/chords/chords.data", "r")
chords = {}
for chord in chords_dataset:
    chord_arr = chord.split(",")
    chords["".join(list(map(lambda x: "1" if x == "YES" else "0", chord_arr[2:14])))] = chord_arr[-1].strip()

print(sys.argv[1].split("/")[-1])
g.readline()
for line in g.readlines():
    try:
        left_hand = line.split(",")[1]
        left_hand_notes_vel = left_hand.split("-")
        left_hand_notes = []
        for left_hand_note_vel in left_hand_notes_vel:
            left_hand_note = left_hand_note_vel.split("|")[0]
            left_hand_notes.append(int(left_hand_note) % 12)

        lhBitMap = ["0"]*12
        for i in range(12):
            if i in left_hand_notes:
                lhBitMap[i] = "1"

        #lhBitMap = lhBitMap[2:]+lhBitMap[:2] #account for C is starting in that Bach dataset.
        chord = "".join(lhBitMap)
        if chord in chords:
            new_file.write(line.strip() + ","+chords[chord] + "\n")
        else:
            while "".join(lhBitMap) not in chords and len(left_hand_notes) != 0:
                left_hand_notes = left_hand_notes[:-1]
                lhBitMap = ["0"] * 12
                for i in range(12):
                    if i in left_hand_notes:
                        lhBitMap[i] = "1"
                #lhBitMap = lhBitMap[2:] + lhBitMap[:2]

            if "".join(lhBitMap) in chords:
                new_file.write(line.strip() + "," + chords["".join(lhBitMap)] + "\n")
            else:
                chord_poss = difflib.get_close_matches(chord, chords.keys(), 1)
                new_file.write(line.strip() + "," + chords[chord_poss[0]] + "\n")


    except ValueError:
        print("Exception occured, Value error")



