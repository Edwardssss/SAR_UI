# Import Module 
from tkinter import *

# Create Object 
root = Tk() 

# Set height and width 
width = 450
height = 300

# Set Geometry and min, max size 
root.geometry(f"{width}x{height}") 
root.maxsize(width, height) 
root.minsize(width, height) 

# Create Label 
Label(root, text="Average Speed Calculator", font=( 
	"Helvetica", 18, "bold"), fg="blue").pack() 

# Calculate Average Speed 


def average_speed_calculator(): 
		# Get the value of spinbox using get() method 
		# Hours 
	hrs = int(hours.get()) 
	# minutes 
	mins = int(minutes.get()) 
	# distance 
	dist = int(distance.get()) 

	# Formule Used 
	value = dist/(hrs+(mins/60)) 

	# change the text of label using config method 
	average_speed.config(text=f"{value} Km/Hr") 


# Create Mulitiple Frames 
frame = Frame(root) 
frame.pack() 

frame1 = Frame(root) 
frame1.pack() 

frame2 = Frame(root) 
frame2.pack() 

# Create Labels and Spin Boxes 
Label(frame, text="Hours", width=15, font=("Helvetica", 14, "bold"), 
	borderwidth=2, relief="solid").pack(side=LEFT, padx=10, pady=10) 
hours = Spinbox(frame, from_=0, to=10000000, width=5, 
				font=("Helvetica", 14, "bold")) 
hours.pack(side=LEFT, pady=10) 

Label(frame1, text="Minutes", width=15, font=("Helvetica", 14, "bold"), 
	borderwidth=2, relief="solid").pack(side=LEFT, padx=10, pady=10) 
minutes = Spinbox(frame1, from_=0, to=10000000, width=5, 
				font=("Helvetica", 14, "bold")) 
minutes.pack(side=LEFT, pady=10) 

Label(frame2, text="Distance (Km)", width=15, font=("Helvetica", 14, "bold"), 
	borderwidth=2, relief="solid").pack(side=LEFT, padx=10, pady=10) 
distance = Spinbox(frame2, from_=0, to=10000000, width=5, 
				font=("Helvetica", 14, "bold")) 
distance.pack(side=LEFT, pady=10) 

Button(root, text="Average Speed", width=15, font=("Helvetica", 14, "bold"), 
	command=average_speed_calculator, fg="red", bg="black").pack(pady=20) 
average_speed = Label(root, text="", width=20, font=( 
	"Helvetica", 14, "bold"), relief="solid") 
average_speed.pack() 

# Execute Tkinter 
root.mainloop() 
