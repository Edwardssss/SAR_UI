from tkinter import *
import time

win = Tk()
win.title("tutorialspoint.com")
win.geometry("600x400")

def clock():
    hh= time.strftime("%I")
    mm= time.strftime("%M")
    ss= time.strftime("%S")
    day=time.strftime("%A")
    ap=time.strftime("%p")
    time_zone= time.strftime("%Z")
    my_lab.config(text= hh + ":" + mm +":" + ss)
    my_lab.after(1000,clock)
    
    my_lab1.config(text=time_zone+" "+ day)

def updateTime():
    my_lab.config(text= "New Text")
    
my_lab= Label(win,text= "",font=("sans-serif", 56), fg= "red")
my_lab.pack(pady=20)
my_lab1= Label(win, text= "", font=("Helvetica",20), fg= "blue")
my_lab1.pack(pady= 10)
clock()

win.mainloop()
