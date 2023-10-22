import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math

# Funcion LOG
def Log():
    print("Hola")

# Nuesto sistema tendra codEstudiante, nombreEstudiante, carrera

# Cargar variables de entorno
load_dotenv()
# Paths
OUT_FOLDER_PATH_USERS = os.getenv("PATH_USERS")
PATH_USER_CHECK = os.getenv("PATH_USERS_CHECK")
OUT_FOLDER_PATH_FACES = os.getenv("PATH_FACES")

# Para manejar info, hacerlo en funciones
info = []

# Interface
pantalla = Tk()
pantalla.title("Kikness | MVP")  # Kinesis
pantalla.configure(bg="white")
pantalla.geometry("1280x720")

# Background
img_background = PhotoImage(file=os.getenv("PATH_BACKGROUND"))
background = Label(image=img_background, text="Inicio")
background.place(x=0, y=0, relheight=1, relwidth=1)

# Manejo de inputs de registro (codEstudiante, nombreEstudiante, profesion)
# codEstudiante
input_cod_estudiante_reg = Entry(pantalla)
input_cod_estudiante_reg.place(x=80, y=220)

# nombresEstudiante
input_nom_estudiante_reg = Entry(pantalla)
input_nom_estudiante_reg.place(x=80, y=330)

# apellidosEstudiante
input_ape_estudiante_reg = Entry(pantalla)
input_ape_estudiante_reg.place(x=80, y=440)

# carreraEstudiante
opcion_seleccionada = StringVar() # Obtener el valor en string
opciones = ["","Ing. Software", "Ing. Industrial", "Ing. Civil", "Administración"]
opcion_seleccionada.set("Seleccione la carrera")
select = OptionMenu(pantalla, opcion_seleccionada, *opciones)
select.pack()
select.place(x=80, y=550)

# btn de registrar
img_register = PhotoImage(file=os.getenv("PATH_BTN_REGISTER"))
btn_register = Button(pantalla, image=img_register, compound="top", command=Log, highlightthickness=0, bd=0, bg="white")
btn_register.place(x=248, y=613)

'''[
{
codEstudiante: u19201322,
nomEstudiante: William Elisban,
apeEstudiante: Chávez Díaz,
carrera: Ing. Software
}
]'''


# Se lanza la pantalla
pantalla.mainloop()
