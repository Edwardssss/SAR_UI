
#Importing the tkinter library
from tkinter import *

#Create an instance of tkinter frame
win= Tk()

#Define the size of the window or frame
win.geometry("600x400")

#To Make it transparent use alpha property to define the opacity of the frame
win.attributes('-alpha', 0.3)
win.mainloop()

