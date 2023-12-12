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
import datetime as dt
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import random
import pandas as pd


def codeFace(images):
    # List
    lista_cod = []

    for img in images:
        # Color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img
        cod = fr.face_encodings(img)[0]
        lista_cod.append(cod)  # save list

    return lista_cod


def closeWindow():
    global step, conteo
    step = 0
    conteo = 0
    pantalla2.destroy()


def closeWindow2():
    global step, conteo
    step = 0
    conteo = 0
    pantalla3.destroy()


# Object detect
def objectDetect(img):
    global glass, hat
    glass = False
    hat = False
    frame = img

    # Clases
    class_name_cap = ['Gafas', 'Gorras', 'Casaca', 'Polo', 'Pantalones', 'Shorts', 'Falda', 'Vestido', 'Mochila',
                      'Zapatos']

    # Inferencia
    resultsCap = model_glass_hat(frame, stream=True, imgsz=640)
    resultsGlass = model_glass(frame, stream=True, imgsz=640)

    def process_results(results, target_cls, flag):
        for res in results:
            for box in res.boxes:
                xi, yi, xf, yf = [int(max(0, coord)) for coord in box.xyxy[0]]

                cls = int(box.cls[0])
                conf = math.ceil(box.conf[0])

                if cls == target_cls:
                    flag = True
                    cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 255, 0), 2)
                    cv2.putText(frame, f"{class_name_cap[cls]} {int(conf * 100)}%", (xi, yi - 20),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255, 0), 2)

        return flag

    hat = process_results(resultsCap, 1, hat)
    glass = process_results(resultsGlass, 0, glass)

    return frame


# Funcion LOG
def createEstudiante():
    # Agregar un regex para validar el codigo de estudiante
    global cod_estudiante, nom_estudiante, ape_estudiante, carrera, cap, lblVideo, pantalla2
    global x, y
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
        print(f"LOG: Registro exitoso {cod_estudiante} -> {dt.datetime.now()}")
    except:
        print(f"LOG: Error en el registro de un estudiante -> {dt.datetime.now()}")

    # Limpiar los campos
    input_cod_estudiante_reg.delete(0, END)
    input_nom_estudiante_reg.delete(0, END)
    input_ape_estudiante_reg.delete(0, END)
    opcion_seleccionada.set("Seleccione la carrera")

    # Se lanza pantalla para registrar un nuevo rostro, en un futuro daremos la opcion de subir la foto directamente
    pantalla2 = Toplevel(pantalla)
    pantalla2.title("Kikness | Registro Biometrico")
    pantalla2.geometry("1280x720")
    pantalla2.resizable(False, False)
    pantalla2.geometry(f"1280x720+{x}+{y}")

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
        frame_object_detect = frame.copy()

        if ret == True:
            # inferencia de malla facial
            res = FaceMesh.process(frame_rgb)

            # Object detect
            # frame = objectDetect(frame_object_detect)

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
                                            # and glass == False and hat == False):
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
                                                frame[155:155 + alcheck, 1105:1105 + ancheck] = img_check

                                                # conteo de parpadeos
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 370),
                                                            0, 0.5, (0, 0, 0), 1)  # cv2.FONT_HERSHEY_COMPLEX

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
                                            # and glass == False and hat == False):
                                            cv2.rectangle(frame, (xi, yi, anch, alt), (0, 255, 0),
                                                          2)  # Color del rectangulo
                                            # img check liveness
                                            alli, anli, c = img_livenesscheck.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_livenesscheck
                                            # messagebox.showinfo("Registro", "Registro satisfactorio")

                                        '''if glass == True:
                                            # Img glasses
                                            al_glass, an_glass = img_glass.shape
                                            frame[50:50 + al_glass, 50:50 + an_glass] = img_glass

                                        if hat == True:
                                            # Img glasses
                                            al_hat, an_hat = img_hat.shape
                                            frame[50:50 + al_hat, 50:50 + an_hat] = img_hat'''

                                    close = pantalla2.protocol("WM_DELETE_WINDOW", closeWindow)
                                    # pantalla2.protocol("WM_DELETE_WINDOW", closeWindow)

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
    global x, y

    # DB Faces
    images = []
    clases = []

    lista = os.listdir(OUT_FOLDER_PATH_FACES)

    # Leer rostros
    for l in lista:
        imgdb = cv2.imread(f"{OUT_FOLDER_PATH_FACES}/{l}")
        # Save Img DB
        images.append(imgdb)
        # name img
        clases.append(os.path.splitext(l)[0])

    # Face Code
    face_code = codeFace(images)
    # print(face_code)

    # pantalla3
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("Kikness | Control de acceso")
    pantalla3.geometry("1280x720")
    pantalla3.resizable(False, False)
    pantalla3.geometry(f"1280x720+{x}+{y}")

    # nuevo lbl de video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # videocaptura
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    validarIdentidad()


def validarIdentidad():
    global OUT_FOLDER_PATH_FACES, cap, lblVideo, pantalla3, face_code, clases, images, step, parpadeo, conteo, cod_estudiante, OUT_FOLDER_PATH_FACES_INTRUSOS
    z = 0
    if cap is not None:
        ret, frame = cap.read()

        frame_save = frame.copy()

        # Redimensionar
        frame = imutils.resize(frame, width=1280)

        # frame rgb
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Frame show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_object_detect = frame.copy()

        if ret == True:
            # inferencia de malla facial
            res = FaceMesh.process(frame_rgb)

            # Object detect
            # frame = objectDetect(frame_object_detect)

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
                                            # and glass == False and hat == False)\

                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anch, alt), (255, 0, 255),
                                                          2)  # Color del rectangulo

                                            # img step 0
                                            als0, ans0, c = img_step.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step
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
                                                frame[155:155 + alcheck, 1105:1105 + ancheck] = img_check

                                                # conteo de parpadeos
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True
                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 370),
                                                            0, 0.5, (0, 0, 0), 1)  # cv2.FONT_HERSHEY_COMPLEX

                                                if conteo >= 3:
                                                    alcheck, ancheck, c = img_check.shape
                                                    frame[385:385 + alcheck, 1105:1105 + ancheck] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 15 and longitud2 > 15:
                                                        # step 1
                                                        step = 1
                                            else:
                                                conteo = 0

                                        # paso 1
                                        if step == 1:
                                            # and glass == False and hat == False):
                                            cv2.rectangle(frame, (xi, yi, anch, alt), (0, 255, 0),
                                                          2)  # Color del rectangulo
                                            # img check liveness - ESTO YA NO SIRVE
                                            # alli, anli, c = img_step2.shape
                                            # frame[50:50 + alli, 50:50 + anli] = img_step2
                                            # messagebox.showinfo("Registro", "Registro satisfactorio")

                                            # Find faces
                                            faces_loc = fr.face_locations(frame_rgb)
                                            faces_cod = fr.face_encodings(frame_rgb, faces_loc)

                                            for faces_cod, faces_loc in zip(faces_cod, faces_loc):
                                                # Matching
                                                Match = fr.compare_faces(face_code, faces_cod)

                                                # Sim
                                                simi = fr.face_distance(face_code, faces_cod)

                                                # Min
                                                min = np.argmin(simi)

                                                z = 0
                                                print(Match[min])
                                                if Match[min]:
                                                    # De aca se saca el id que hace match
                                                    cod_estudiante = clases[min].upper()
                                                    print(clases[min])
                                                    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                                    escribir_acceso_excel(cod_estudiante, timestamp)
                                                    closeWindow2()
                                                    profile()  # aca mandarias el websocket, para la validacion en back, para ello necesitamos el endpoint
                                                    z = 1
                                                else:
                                                    # cut
                                                    cut = frame_save[yi:yf, xi:xf]
                                                    # guardamos el rostro
                                                    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                                                    cv2.imwrite(
                                                        f"{OUT_FOLDER_PATH_FACES_INTRUSOS}/intruso_{timestamp}.png",
                                                        cut)
                                                    print(
                                                        f"LOG: No hay patron de reconocimiento, posible intento de suplantación -> {dt.datetime.now()}")
                                                    messagebox.showinfo("Acceso denegado",
                                                                        "Posible intento de suplantación")
                                                    closeWindow2()
                                                    torre = random.choice(['A', 'B'])
                                                    lectora = random.randint(1, 2)
                                                    escribir_intrusos_excel(torre, lectora, timestamp)
                                                    alerta_intruso(torre, lectora, timestamp)
                                                    return

                                        '''if glass == True:
                                            # Img glasses
                                            al_glass, an_glass, c = img_glass.shape
                                            frame[50:50 + al_glass, 50:50 + an_glass] = img_glass

                                        if hat == True:
                                            # Img glasses
                                            al_hat, an_hat, c = img_hat.shape
                                            frame[50:50 + al_hat, 50:50 + an_hat] = img_hat'''

                                        # print(user_name)

                                    # close = pantalla3.protocol("WM_DELETE_WINDOW", closeWindow2)
                                    # pantalla2.protocol("WM_DELETE_WINDOW", closeWindow)

                                    # Circle - Solo es de prueba
                            # cv2.circle(frame, (x1, y1), 2, (255, 0, 0), cv2.FILLED)
                            # cv2.circle(frame, (x2, y2), 2, (255, 0, 0), cv2.FILLED)

        # Convertir el video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Mostrar el video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(5, validarIdentidad)
        if z == 1: closeWindow2()
    else:
        cap.release()


def profile():
    global step, conteo, user_name, OUT_FOLDER_PATH_USERS, cod_estudiante
    global x, y

    # Reset variables
    step = 0
    conteo = 0

    # pantalla4
    pantalla4 = Toplevel(pantalla)
    pantalla4.title("Kikness | Perfil")
    pantalla4.geometry("1280x720")
    pantalla4.resizable(False, False)
    pantalla4.geometry(f"1280x720+{x}+{y}")

    # Fondo
    bc = Label(pantalla4, image=imagenbc, text="Perfil")
    bc.place(x=0, y=0, relheight=1, relwidth=1)

    # File
    user_file = open(f"{OUT_FOLDER_PATH_USERS}/{cod_estudiante}.txt", "r")
    info_user = user_file.read().split(',')
    cod_user, name_user, last_user, facultad = info_user[:4]

    print(f"LOG: Ingreso {cod_user} {name_user} {last_user} -> {dt.datetime.now()}")

    # check user
    if cod_user in clases:
        codigo_estudiante = Label(pantalla4, text=f"{cod_user}", font=("Arial", 15, "bold"),
                                  bg="white")  # Cambiar el tamaño, negrita y fuente
        codigo_estudiante.place(x=850, y=233)
        nombres_estudiante = Label(pantalla4, text=f"{name_user}", font=("Arial", 15, "bold"), bg="white")
        nombres_estudiante.place(x=850, y=305)
        apellidos_estudiante = Label(pantalla4, text=f"{last_user}", font=("Arial", 15, "bold"), bg="white")
        apellidos_estudiante.place(x=850, y=380)
        facultad = Label(pantalla4, text=f"{facultad}", font=("Arial", 15, "bold"), bg="white")
        facultad.place(x=850, y=455)

        # Profile photo - saldremos feos, ya que no estoy registrando foto de presentacion
        lbl_image = Label(pantalla4, bg="white")
        lbl_image.place(x=200, y=200)

        # OUT_FOLDER_PATH_FACES OUT_FOLDER_PATH_PROFILES
        # rostro = OUT_FOLDER_PATH_FACES # Original path
        rostro = OUT_FOLDER_PATH_PROFILES
        img_user = cv2.imread(f"{rostro}/{cod_user}.png")
        img_user = cv2.cvtColor(img_user, cv2.COLOR_RGB2BGR)  # fijate en la conversion de color
        img_user = Image.fromarray(img_user)

        nuevo_ancho = 300  # Nuevo ancho deseado
        nuevo_alto = 340  # Nuevo alto deseado
        img_user = img_user.resize((nuevo_ancho, nuevo_alto))

        IMG = ImageTk.PhotoImage(image=img_user)

        lbl_image.configure(image=IMG)
        lbl_image.image = IMG


def alerta_intruso(torre, lectora, timestamp):
    global OUT_FOLDER_PATH_FACES_INTRUSOS
    email_address = 'chavezdiaz4@gmail.com'
    recipient_emails = ', '.join(['chavezdiaz4@gmail.com', 'kikness_test@yopmail.com'])
    try:
        # Configurar el mensaje
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = recipient_emails  # Concatena los correos con comas
        msg['Subject'] = 'Alerta posible suplantación de identidad'

        fecha_hora = dt.datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
        fecha = fecha_hora.strftime("%Y-%m-%d")
        hora = fecha_hora.strftime("%H:%M:%S")

        # Crear el contenido HTML personalizado
        # HTML con estilos en línea para la negrita del detalle
        html = f"""
            <html>
                <head>
                    <style>
                        .bold-text {{
                            font-weight: bold;
                        }}
                    </style>
                </head>
                <body style="background-color: white; color: black; font-family: Arial, sans-serif;">
                    <div style="background-color: #00C434; padding: 20px;">
                        <h1 style="color: white;">Kikness | One trait at a time</h1>
                    </div>
                    <div style="padding: 20px;">
                        <p>Estimado usuario,</p>
                        <p>Se ha detectado un intento de Suplantación de identidad.</p>
                        <p class="bold-text">Detalles:</p>
                        <p><span class="bold-text">Torre:</span> {torre}</p>
                        <p><span class="bold-text">Lectora:</span> {lectora}</p>
                        <p><span class="bold-text">Fecha:</span> {fecha}</p>
                        <p><span class="bold-text">Hora:</span> {hora}</p>
                    </div>
                    <div style="background-color: #00C434; color: white; padding: 10px; display: flex; justify-content: space-between;">
                        <div>© AI 2023</div>
                    </div>
                </body>
            </html>
        """

        # Agregar el cuerpo del mensaje
        # msg.attach(MIMEText(
        #   f'No hay patron de reconocimiento, posible intento de suplantación -> {dt.datetime.now()}\nPor favor, acercarse a verificar a la puerta de ingreso de la universidad.',
        #  'plain'))

        msg.attach(MIMEText(html, 'html'))

        # Adjuntar la imagen
        img_path = f"{OUT_FOLDER_PATH_FACES_INTRUSOS}intruso_{timestamp}.png"  # Ruta de la imagen que quieres adjuntar
        with open(img_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(img_path))
            msg.attach(img)

        # Conectar al servidor SMTP
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email_address, os.getenv("PASS"))
        text = msg.as_string()
        server.sendmail(email_address, recipient_emails, text)
        print('¡Correo enviado correctamente!')
    except Exception as e:
        print('Error al enviar el correo:', e)
    finally:
        server.quit()


def escribir_acceso_excel(cod_estudiante, timestamp):
    ruta_guardado = 'C:/Users/willi/Desktop/IA_Project_Autenticacion/database/reports/control_accesos.xlsx'

    fecha_hora = dt.datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    fecha_acceso = fecha_hora.strftime("%Y-%m-%d")
    hora_acceso = fecha_hora.strftime("%H:%M:%S")

    user_file = open(f"{OUT_FOLDER_PATH_USERS}/{cod_estudiante}.txt", "r")
    info_user = user_file.read().split(',')
    cod_user, name_user, last_user, facultad = info_user[:4]

    # Crear un diccionario con los datos recibidos
    data = {
        'cod_estudiante': [cod_estudiante],
        'nombres': [name_user],
        'apellidos': [last_user],
        'facultad': [facultad],
        'hora_acceso': [hora_acceso],
        'fecha_acceso': [fecha_acceso]
    }

    # Intentar cargar el archivo existente o crear uno nuevo si no existe
    try:
        # Si el archivo existe, cargarlo en un DataFrame
        df = pd.read_excel(ruta_guardado)
        # Crear un DataFrame con los nuevos datos
        nuevo_registro = pd.DataFrame(data)
        # Agregar nuevos datos al DataFrame existente
        df = pd.concat([df, nuevo_registro], ignore_index=True)
    except FileNotFoundError:
        # Si el archivo no existe, crear uno nuevo con los datos
        df = pd.DataFrame(data)

    # Escribir los datos en el archivo Excel en la ruta proporcionada
    df.to_excel(ruta_guardado, index=False)


def escribir_intrusos_excel(torre, lectora, timestamp):
    ruta_guardado = 'C:/Users/willi/Desktop/IA_Project_Autenticacion/database/reports/control_intrusos.xlsx'

    fecha_hora = dt.datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")
    fecha_acceso = fecha_hora.strftime("%Y-%m-%d")
    hora_acceso = fecha_hora.strftime("%H:%M:%S")

    # Crear un diccionario con los datos recibidos
    data = {
        'cod_registro': [f'{timestamp}_{torre}_{lectora}'],
        'torre': [torre],
        'lectora': [lectora],
        'hora_registro': [hora_acceso],
        'fecha_registro': [fecha_acceso]
    }

    # Intentar cargar el archivo existente o crear uno nuevo si no existe
    try:
        # Si el archivo existe, cargarlo en un DataFrame
        df = pd.read_excel(ruta_guardado)
        # Crear un DataFrame con los nuevos datos
        nuevo_registro = pd.DataFrame(data)
        # Agregar nuevos datos al DataFrame existente
        df = pd.concat([df, nuevo_registro], ignore_index=True)
    except FileNotFoundError:
        # Si el archivo no existe, crear uno nuevo con los datos
        df = pd.DataFrame(data)

    # Escribir los datos en el archivo Excel en la ruta proporcionada
    df.to_excel(ruta_guardado, index=False)


# Nuesto sistema tendra codEstudiante, nombreEstudiante, carrera

# Cargar variables de entorno
load_dotenv()
# Paths
OUT_FOLDER_PATH_USERS = os.getenv("PATH_USERS")
PATH_USER_CHECK = os.getenv("PATH_USERS_CHECK")
OUT_FOLDER_PATH_FACES = os.getenv("PATH_FACES")
OUT_FOLDER_PATH_PROFILES = os.getenv("PATH_PROFILES")
OUT_FOLDER_PATH_FACES_INTRUSOS = os.getenv("PATH_FACES_INTRUSOS")

# Modelos
model_glass_hat = YOLO('modelos/ModeloGafasGorras.pt')
model_glass = YOLO('modelos/ModeloGafas.pt')

# Read Img
img_check = cv2.imread(os.getenv("PATH_ICON_CHECK"))
img_check = cv2.cvtColor(img_check, cv2.COLOR_RGB2BGR)
img_step = cv2.imread(os.getenv("PATH_ICON_MVP"))  # mvp
img_step = cv2.cvtColor(img_step, cv2.COLOR_RGB2BGR)
img_step0 = cv2.imread(os.getenv("PATH_ICON_STEP0"))  # Step 0
img_step0 = cv2.cvtColor(img_step0, cv2.COLOR_RGB2BGR)
img_step1 = cv2.imread(os.getenv("PATH_ICON_STEP1"))  # Step 1
img_step1 = cv2.cvtColor(img_step1, cv2.COLOR_RGB2BGR)
img_step2 = cv2.imread(os.getenv("PATH_ICON_STEP2"))  # Step 2
img_step2 = cv2.cvtColor(img_step2, cv2.COLOR_RGB2BGR)
img_livenesscheck = cv2.imread(os.getenv("PATH_LIVENESS_CHECK"))  # LivenessCheck
img_livenesscheck = cv2.cvtColor(img_livenesscheck, cv2.COLOR_RGB2BGR)
img_glass = cv2.imread("C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/glases.png")
img_hat = cv2.imread("C:/Users/willi/Desktop/IA_Project_Autenticacion/assets/cap.png")

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

# Obtiene el ancho y alto de la pantalla
ancho_pantalla = pantalla.winfo_screenwidth()
alto_pantalla = pantalla.winfo_screenheight()
# Calcula las coordenadas para centrar la ventana en la pantalla
x = (ancho_pantalla - 1280) // 2  # 1280 es el ancho de la ventana
y = (alto_pantalla - 720) // 2  # 720 es la altura de la ventana

# Establece la ubicación de la ventana en el centro de la pantalla
pantalla.geometry(f"1280x720+{x}+{y}")

# Background
img_background = PhotoImage(file=os.getenv("PATH_BACKGROUND"))
background = Label(image=img_background, text="Inicio")
background.place(x=0, y=0, relheight=1, relwidth=1)

# Profile Background
imagenbc = PhotoImage(file=os.getenv("PATH_BACKGROUND_PROFILE"))

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
opciones = ["", "Arquitectura", "Ciencias de la Salud", "Comunicaciones", "Derecho", "Educación", "Ingeniería",
            "Negocios", "Psicología"]
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
