import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from dotenv import load_dotenv
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import imutils
import math
import re


def closeWindow():
    global step, conteo
    step = 0
    conteo = 0
    pantalla2.destroy()


# Funcion LOG
def createEstudiante():
    # Agregar un regex para validar el codigo de estudiante
    global cod_estudiante, nom_estudiante, ape_estudiante, carrera, cap, lblVideo, pantalla2
    cod_estudiante = input_cod_estudiante_reg.get()
    nom_estudiante = input_nom_estudiante_reg.get()
    ape_estudiante = input_ape_estudiante_reg.get()
    carrera = opcion_seleccionada.get()

    vacio = [len(cod_estudiante), len(nom_estudiante), len(ape_estudiante), len(carrera)]

    if 0 in vacio:
        messagebox.showinfo("Alerta", "No pueden haber campos vacios")
        return

    regex = r'^U\d{8}$'
    match = re.match(regex, cod_estudiante)

    if match is None:
        messagebox.showinfo("Alerta", "El codigo de estudiante no sigue la nomenclatura UXXXXXXXX")
        return

    estudiantes_registrados = os.listdir(PATH_USER_CHECK)

    codigos_estudiantes = []

    for estudiante in estudiantes_registrados:
        estudiante = estudiante.split('.')
        codigos_estudiantes.append(estudiante[0])  # Guardamos codigo

    # Verificamos registro de estudiante
    if cod_estudiante in codigos_estudiantes:
        messagebox.showinfo("Alerta", "El usuario ha sido registrado anteriormente")
        return

    # Registramos al estudiante
    info.append(cod_estudiante)
    info.append(nom_estudiante)
    info.append(ape_estudiante)
    info.append(carrera)

    # save, en un futuro manejalo como un json localmente, para la tesis esto ira en bd
    try:
        f = open(f"{OUT_FOLDER_PATH_USERS}/{cod_estudiante}.txt", "w")
        f.write(f"{cod_estudiante},{nom_estudiante},{ape_estudiante},{carrera}")
        f.close()
        print("Registro exitoso")
    except:
        print("Error en el registro de un estudiante")

    # Limpiar los campos
    input_cod_estudiante_reg.delete(0, END)
    input_nom_estudiante_reg.delete(0, END)
    input_ape_estudiante_reg.delete(0, END)
    opcion_seleccionada.set("Seleccione la carrera")

    # Se lanza pantalla para registrar un nuevo rostro, en un futuro daremos la opcion de subir la foto directamente
    pantalla2 = Toplevel(pantalla)
    pantalla2.title("Kikness | Registro Biometrico")
    pantalla2.geometry("1280x720")
    # Creamos el label de video, osea usaremos nuestra camara
    lblVideo = Label(pantalla2)
    lblVideo.place(x=0, y=0)

    # videocaptura
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    registroBiometrico()


# En esta funcion se registrara todo
def registroBiometrico():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo, cod_estudiante

    if cap is not None:
        ret, frame = cap.read()

        frame_save = frame.copy()

        # Redimensionar
        frame = imutils.resize(frame, width=1280)

        # frame rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Frame show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            # inferencia de malla facial
            res = FaceMesh.process(frame_rgb)

            # lista de resultados
            px = []
            py = []
            lista_coordenadas = []

            if res.multi_face_landmarks:
                # Extraer detecciones de malla facial
                for rostros in res.multi_face_landmarks:
                    # dibujamos malla facial
                    mp_draw.draw_landmarks(frame, rostros, face_mesh_object.FACEMESH_CONTOURS, config_draw,
                                           config_draw)  # Tambien puede ser FACEMESH_TESSELATION

                    # Extraemos los puntos
                    for id, puntos in enumerate(rostros.landmark):
                        # info img
                        alto, ancho, c = frame.shape
                        x, y = int(puntos.x * ancho), int(puntos.y * alto)
                        px.append(x)
                        py.append(y)
                        lista_coordenadas.append([id, x, y])
                        # puntos clave: los de las cejas, los de la cien, asi sabemos si esta mirando de frente o no, me quede en 1:20
                        # 468 KeyPoints
                        if len(lista_coordenadas) == 468:
                            # ojo derecho
                            x1, y1 = lista_coordenadas[145][1:]
                            x2, y2 = lista_coordenadas[159][1:]
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            # ojo izq
                            x3, y3 = lista_coordenadas[374][1:]
                            x4, y4 = lista_coordenadas[386][1:]
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            # parietal derecho
                            x5, y5 = lista_coordenadas[139][1:]

                            # parietal izq
                            x6, y6 = lista_coordenadas[368][1:]

                            # ceja derecha
                            x7, y7 = lista_coordenadas[70][1:]

                            # ceja izq
                            x8, y8 = lista_coordenadas[300][1:]

                            # deteccion de rostros
                            faces = detector.process(frame_rgb)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    # Bbox: recuadro del rostros - "ID, BBOX, SCORE
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > conf_threshold:
                                        # convertir a pixeles
                                        xi, yi, anch, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anch, alt = int(xi * ancho), int(yi * alto), int(anch * ancho), int(
                                            alt * alto)

                                        # Offset X
                                        offset_anch = (offset_x / 100) * anch
                                        xi = int(xi - int(offset_anch / 2))
                                        anch = int(anch + offset_anch)
                                        xf = xi + anch

                                        # Offset Y
                                        offset_alt = (offset_y / 100) * alt
                                        yi = int(yi - offset_alt)
                                        alt = int(alt + offset_alt)
                                        yf = yi + alt

                                        # Control de error sin rostros
                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if anch < 0: anch = 0
                                        if alt < 0: alt = 0

                                        # Steps - pasos de verificacion
                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anch, alt), (255, 0, 255),
                                                          2)  # Color del rectangulo

                                            # img step 0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0
                                            # img step 1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1
                                            # img step 3
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Aca verificamos que este mirando a la camara
                                            if x7 > x5 and x8 < x6:  # Estamos mirando hacia el frente
                                                # img check
                                                alcheck, ancheck, c = img_check.shape
                                                frame[165:165 + alcheck, 1105:1105 + ancheck] = img_check

                                                # conteo de parpadeos
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),
                                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                                if conteo >= 3:
                                                    alcheck, ancheck, c = img_check.shape
                                                    frame[385:385 + alcheck, 1105:1105 + ancheck] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 15 and longitud2 > 15:
                                                        # cut
                                                        cut = frame_save[yi:yf, xi:xf]
                                                        # guardamos el rostro
                                                        cv2.imwrite(f"{OUT_FOLDER_PATH_FACES}/{cod_estudiante}.png",
                                                                    cut)

                                                        # step 1
                                                        step = 1
                                            else:
                                                conteo = 0

                                        # paso 1
                                        if step == 1:
                                            cv2.rectangle(frame, (xi, yi, anch, alt), (0, 255, 0),
                                                          2)  # Color del rectangulo
                                            # img check liveness
                                            alli, anli, c = img_livenesscheck.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_livenesscheck

                                    close = pantalla2.protocol("WM_DELETE_WINDOW", closeWindow)

                                    # Circle - Solo es de prueba
                            # cv2.circle(frame, (x1, y1), 2, (255, 0, 0), cv2.FILLED)
                            # cv2.circle(frame, (x2, y2), 2, (255, 0, 0), cv2.FILLED)

        # Convertir el video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Mostrar el video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(5, registroBiometrico)
    else:
        cap.release()


# Funcion MVP
def lanzarMVP():
    global OUT_FOLDER_PATH_FACES, cap, lblVideo, pantalla3, face_code, clases, images



# Nuesto sistema tendra codEstudiante, nombreEstudiante, carrera

# Cargar variables de entorno
load_dotenv()
# Paths
OUT_FOLDER_PATH_USERS = os.getenv("PATH_USERS")
PATH_USER_CHECK = os.getenv("PATH_USERS_CHECK")
OUT_FOLDER_PATH_FACES = os.getenv("PATH_FACES")

# Read Img
img_check = cv2.imread('C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/check.png')
img_step0 = cv2.imread('C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/Step0.png')
img_step1 = cv2.imread('C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/Step1.png')
img_step2 = cv2.imread('C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/Step2.png')
img_livenesscheck = cv2.imread('C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/LivenessCheck.png')

# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

# Offset
offset_y = 40
offset_x = 20

# Threshold -> umbral de confianza de deteccion
conf_threshold = 0.5

# tool draw -> malla facial
mp_draw = mp.solutions.drawing_utils
config_draw = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

# Object de malla facial
face_mesh_object = mp.solutions.face_mesh
FaceMesh = face_mesh_object.FaceMesh(max_num_faces=1)  # Aca se determina cuantos rostros observara mediapipe

# Object de detector de rostros
face_object = mp.solutions.face_detection
detector = face_object.FaceDetection(min_detection_confidence=0.5, model_selection=1)  # umbral de deteccion

# Para manejar info, hacerlo en funciones
info = []

# Interface
pantalla = Tk()
pantalla.title("Kikness | MVP")  # Kinesis
pantalla.configure(bg="white")
pantalla.geometry("1280x720")
pantalla.resizable(False, False)

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
opcion_seleccionada = StringVar()  # Obtener el valor en string
opciones = ["", "Ing. Software", "Ing. Industrial", "Ing. Civil", "Administración"]
opcion_seleccionada.set("Seleccione la carrera")
select = OptionMenu(pantalla, opcion_seleccionada, *opciones)
select.pack()
select.place(x=80, y=550)

# btn de registrar
img_register = PhotoImage(file=os.getenv("PATH_BTN_REGISTER"))
btn_register = Button(pantalla, image=img_register, compound="top", command=createEstudiante, highlightthickness=0,
                      bd=0, bg="white")
btn_register.place(x=248, y=613)

# btn para lanzar MVP
img_mvp = PhotoImage(file=os.getenv("PATH_BTN_MVP"))
btn_mvp = Button(pantalla, image=img_mvp, compound="top", command=lanzarMVP, highlightthickness=0, bd=0, bg="white")
btn_mvp.place(x=877, y=613)

'''
Arreglo que contiene otros JSON

[
{
codEstudiante: u19201322,
nomEstudiante: William Elisban,
apeEstudiante: Chávez Díaz,
carrera: Ing. Software
}
]'''

# Se lanza la pantalla
pantalla.mainloop()
