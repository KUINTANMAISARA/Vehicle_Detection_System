import sys
print(sys.executable)

import tkinter as tk
from tkinter import filedialog, Text, Label
import tkinter.font as tkFont
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pygame

pygame.mixer.init()
new_model = tf.keras.models.load_model('saved_model\my_model')
pygame.mixer.music.load("CarSound.mp3")
pygame.mixer.music.play() 

batch_size = 32
img_height = 300
img_width = 300

class_names = ['bike', 'campervan', 'supercar', 'suv', 'truck']

root = tk.Tk()

root.title('PROJECT VEHICLE RECOGNITION!')
root.geometry("700x650")

filename ="null"


apps = []

def addApp():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",title="Select File",
    filetypes= (("all files","*.*"),("exe","*.exe")))
    apps.append(filename)
    for app in apps:
        label = tk.Label(frame,text=app,bg="white" ) 
        label.pack() 
    return filename

def runApps():
    global filename
    img = keras.preprocessing.image.load_img(
    filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    Output = (
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    labeloutput = tk.Label(frame,text=Output,bg="white")
    labeloutput.pack()


canvas = tk.Canvas(root,height=700, width=700,bg="#263D42")

canvas.pack()

frame =tk.Frame(root,bg="white")
frame.place(relwidth=0.8,relheight=0.8,relx=0.1,rely=0.1)

Cars_image =ImageTk.PhotoImage(Image.open('cargif.gif'))

fontStyle = tkFont.Font(family="Lucida Grande",size=20)
labeltitle = Label(frame,text="PROJECT VEHICLE DETECTION" ,font= fontStyle,bg="white")
labeltitle.pack()

labelproject = Label(frame ,image=Cars_image)
labelproject.pack()

line = tk.Frame(frame, height=1, width=550, bg="grey80", relief='groove')
line.pack()

openFile = tk.Button(frame,text="Open File",padx=10,pady=5,fg="white",bg="#263D42",command=addApp)
openFile.pack()
runApps = tk.Button(frame,text="Run Apps",padx=10,pady=5,fg="white",bg="#263D42",command=runApps)
runApps.pack()

root.mainloop()
