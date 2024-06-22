from tkinter import *
win =Tk()
win.geometry("300x200")
label= Label(win, text= "Select any One Language!", font= ("", 10))
label.pack(pady=30)


clicked= StringVar()
#Create an instance of Menu in the frame
main_menu= OptionMenu(win, clicked, "C++", "Java", "Python", "Rust","Go","Ruby")
main_menu.pack()

win.mainloop()
