# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:39:10 2021

@author: Carlos
"""

from tkinter import *

#%% ejemplo gui
# ventana
root = Tk()

# creat label widget
myLabel = Label(root, text = 'Hello World!')

# empaquetarlo en la pantalla
myLabel.pack()

# loop que mantiene corriendo el programa
root.mainloop()

#%% ejemplo grid

# --------------------
#   crear las cosas
# ---------------------
# ventana
root = Tk()

# creat label widget
myLabel1 = Label(root, text = 'Hello World!')
myLabel2 = Label(root, text = 'My name is Carlos Calvo')
myLabel3 = Label(root, text = '                   ')

# --------------------------
#  ponerlas en la pantalla
# --------------------------
# empaquetarlo en la grilla
myLabel1.grid(row = 0, column = 0)
myLabel2.grid(row = 1, column = 5)
myLabel3.grid(row = 1, column = 1)

# loop que mantiene corriendo el programa
root.mainloop()

#%%  ejemplo botones

# ventana
root = Tk()

def myClick():
    myLabel = Label(root, text = 'Look! I clicked a button!!')
    myLabel.pack()
    
# botón
# myButton = Button(text = 'My button', state = DISABLED)
myButton = Button(text = 'My button', padx = 50, pady = 50, command = myClick, fg = 'blue', bg = 'red')

# ponerlo en la pantalla
myButton.pack()


# loop que mantiene corriendo el programa
root.mainloop()


#%% ejemplo inputs

# ventana
root = Tk()

# entrada
entry = Entry(root, width = 50, bg = 'white', borderwidth = 3)

# mostrarlo
entry.pack()

def myClick():
    myLabel = Label(root, text = 'Hola '+entry.get())
    myLabel.pack()
    
# botón
# myButton = Button(text = 'My button', state = DISABLED)
myButton = Button(text = 'Enter your stock quote', padx = 50, pady = 50, command = myClick, fg = 'blue')

# ponerlo en la pantalla
myButton.pack()


# loop que mantiene corriendo el programa
root.mainloop()


