# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:18:21 2021

@author: Carlos
"""

from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title('Imagenes')


# cargar la imagen
my_img = ImageTk.PhotoImage(Image.open('images.png'))
my_label = Label(image = my_img)
my_label.pack()

# loop que mantiene corriendo el programa
root.mainloop()