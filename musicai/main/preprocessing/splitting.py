import sys
f = open(sys.argv[1])

tempo = int(f.readline().strip().split(",")[1])

bar_length = (60.0/(tempo/4.0)) * 1000 # in milliseconds

g = open("./Formatted/" + sys.argv[1] + ".formatted","w")

header = "Tempo=" + str(tempo) + ",BarLength=" + str(bar_length)
g.write("Right Hand Notes, Left Hand Notes," + header + "\n")


SPLIT_POINT = 60
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
			note_with_intensity = str(l[0]) + "|" + str(l[1]) 
			if l[0] < SPLIT_POINT:
				left.append(note_with_intensity)
			else:
				right.append(note_with_intensity)
		g.write("-".join(right) + "," + "-".join(left) + "\n")		
		bar = [line]
		bar_count += 1
		
f.close()
g.close()


