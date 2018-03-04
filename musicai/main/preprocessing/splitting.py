import sys
from musicai.main.constants.directories import *
f = open(sys.argv[1])

tempo = int(f.readline().strip().split(",")[1])

bar_length = (60.0/(tempo/4.0)) * 1000 # in milliseconds

g = open(PROCESSED + os.path.basename(sys.argv[1]) + ".formatted","w")

header = "Tempo=" + str(tempo) + ",BarLength=" + str(bar_length)
g.write("Right Hand Notes, Left Hand Notes," + header + "\n")


SPLIT_POINT = 62
bar_count = 1
bars = []

bar = []
for line in f:
	line = list(map(int,line.strip().split(",")))
	
	if(line[2] < bar_count * bar_length):
		bar.append(line)
	else:
		left = []
		right = []
		for l in bar:
			note_details = str(l[0]) + "|" + str(l[1]) + "|" + str(l[2]) 
			if l[0] < SPLIT_POINT:
				left.append(note_details)
			else:
				right.append(note_details)
		g.write("-".join(right) + "," + "-".join(left) + "\n")		
		bar = [line]
		bar_count += 1
		
f.close()
g.close()


