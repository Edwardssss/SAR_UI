# Import required Libraries
from tkinter import *
from tkinter import filedialog
import img2pdf
from tkinter import ttk

# Create an instance of tkinter frame
win = Tk()

#set the geometry
win.geometry('600x250')
win.title("Image to PDF")

def select_file():
    global images
    images = filedialog.askopenfilenames(initialdir = "",title = "Select Images")
    Label(win, text=images).pack()

#Convert Image to PDF
def image_to_pdf():
    for image in enumerate(images):
        with open(f"{image}.pdf", "wb") as file:
            file.write(img2pdf.convert(images))
            Label(frame,text=file).pack()

# Add Labels and Buttons
Label(win, text = "Image to PDF Convertor",font = "Caveat 25 bold").pack(pady = 30)
ttk.Button(win, text = "Select Images",command = select_file).pack(ipadx = 10)

frame = Frame(win)
frame.pack()

ttk.Button(frame, text = "Convert and Save",command = image_to_pdf).pack(side = LEFT, pady=20,ipadx = 10)

win.mainloop()
