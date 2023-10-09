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
pantalla.title("Kinesis | MVP")
pantalla.geometry("1280x720")

# Background
img_background = PhotoImage(file=os.getenv("PATH_BACKGROUND"))
background = Label(image=img_background, text="Inicio")
background.place(x=0, y=0, relheight=1, relwidth=1)

# Se lanza la pantalla
pantalla.mainloop()
