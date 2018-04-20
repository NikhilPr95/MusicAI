from tkinter import *
import tkinter

tempo = 0
def getText():
	global tempo
	tempo = E1.get()
	print(tempo)


App = tkinter.Tk()
frame = tkinter.Frame(App)
App.title("MusicAI")

App.resizable(0, 0)
App.config(bg="#ffffff")


L1 = Label(App, text="Tempo").grid(row=0, columnspan=128)
E1 = Entry(App, width=128)
E1.grid(row=1, columnspan=128)
tkinter.Button(App, text="Submit", command=getText).grid(row=1, column=31)
keys = [
	
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
	"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",

]
varColumn = 0
buttons = []

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



def fooForBlackKeys(key):
	# highlight for one second the key
	print(buttons[key])
	buttons[key].config(bg="#00ff00", fg="#000000")
	App.after(500, lambda: buttons[key].config(bg="#000000", fg="#ffffff"))


def fooForWhiteKeys(key):
	# highlight for one second the key
	print(buttons[key])
	buttons[key].config(bg="#00ff00", fg="#ffffff")
	App.after(500, lambda: buttons[key].config(bg="#ffffff", fg="#000000"))

notes = ["C","F","G"]
for i in notes:
	buttons[keys.index(i)].invoke()
	

#App.mainloop()

