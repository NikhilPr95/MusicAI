from tkinter import *
import tkinter
import SharedArray as sa

def getText():
	global tempo,AppInitial
	tempo = E1.get()
	tempo = int(tempo)
	queueForTempo[0] = tempo
	#AppInitial.withdraw()
	#AppInitial.quit()
	AppInitial.destroy()
	createKeyBoard()

tempo = 0

AppInitial = tkinter.Tk()
AppInitial.title("MusicAI")

App = tkinter.Tk()
App.resizable(0, 0)
App.config(bg="#ffffff")

App.title("MusicAI")
App.withdraw()


queueForTempo = []
#create a queue for passing the tempo
if(bytes('tempo','utf-8') in [list(a)[0] for a in sa.list()]):
	sa.delete("shm://tempo")
queueForTempo = sa.create("shm://tempo", 1)
queueForTempo[0] = 0.0

AppInitial.resizable(0, 0)
AppInitial.config(bg="#000000")
AppInitial.geometry("400x200")

L1 = Label(AppInitial, text="Enter the tempo",bg="black",fg="white", font=("Helvetica", 16)).pack()
E1 = Entry(AppInitial, width=40, font = "Calibri 18",justify="center",bg="#1E6FBA",fg="black",disabledbackground="#1E6FBA",disabledforeground="green",highlightbackground="black",highlightcolor="green",highlightthickness=2,bd=0)
E1.pack()
tkinter.Button(AppInitial, text="Click to start playing!", command=getText,width=15,height=1, activebackground='navy',bg='white',fg='black',bd=8,font=('helvetica', 16, 'italic')).pack()
buttons = []
keys = [
	
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",

]

def createKeyBoard():
	global App
	AppInitial.quit()
	varColumn = 0
	
	for key in keys:
	
		if "#" not in key:
			command = lambda x=len(buttons): fooForWhiteKeys(x)
			button = tkinter.Button(App, width=2, height=20, bd=0, bg="#ffffff", fg="#000000", relief="flat", command=command, pady=0)
			buttons.append(button)
			button.grid(row=2, column=varColumn, sticky=tkinter.N+tkinter.E+tkinter.W+tkinter.S)
		
		else:
			command = lambda x=len(buttons): fooForBlackKeys(x)
			button = tkinter.Button(App, width=1, height=10, bd=0, bg="#000000", fg="#ffffff", relief="flat", command=command,pady=0)
			buttons.append(button)
			button.grid(row=2, column=varColumn, ipady=0, sticky=tkinter.N+tkinter.E+tkinter.W)
		varColumn += 1
	
	App.deiconify()
	App.update()


def fooForBlackKeys(key):
	# highlight for one second the key
	print(buttons[key])
	buttons[key].config(bg="#00ff00", fg="#000000")
	App.after(1000*(60/tempo), lambda: buttons[key].config(bg="#000000", fg="#ffffff"))


def fooForWhiteKeys(key):
	# highlight for one second the key
	print(buttons[key])
	buttons[key].config(bg="#00ff00", fg="#ffffff")
	App.after(1000*(60/tempo), lambda: buttons[key].config(bg="#ffffff", fg="#000000"))
'''
notes = ["C","F","G"]
for i in notes:
	buttons[keys.index(i)].invoke()
'''



AppInitial.mainloop()
#App.mainloop()

