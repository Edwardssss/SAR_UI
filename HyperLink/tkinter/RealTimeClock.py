from Tkinter import *
import datetime

root = Tk()

lab = Label(root)
lab.pack()

def clock():
    time = datetime.datetime.now().strftime("Time: %H:%M:%S")
    lab.config(text=time)
    #lab['text'] = time
    root.after(1000, clock) # run itself again after 1000 ms

# run first time
clock()

root.mainloop()
