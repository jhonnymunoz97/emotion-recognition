from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import subprocess


detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

EMOTIONS = ["Enojado","","","Alegre","Triste","","Neutral"]


def procesarDatos(img):
	img = imutils.resize(img,width=300)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
	canvas = np.zeros((250, 300, 3), dtype="uint8")
	frameClone = img.copy()
	
	preds = ""
	cont = 0

	if len(faces) > 0:
		faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
		(fX, fY, fW, fH) = faces
		roi = gray[fY:fY + fH, fX:fX + fW]
		roi = cv2.resize(roi, (64, 64))
		roi = roi.astype("float") / 255.0
		roi = img_to_array(roi)
		roi = np.expand_dims(roi, axis=0)
		preds = emotion_classifier.predict(roi)[0]
		emotion_probability = np.max(preds)
		label = EMOTIONS[preds.argmax()]
	
	for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
		text = "{}: {:.2f}%".format(emotion, prob * 100)
		w = int(prob * 300)
		cv2.rectangle(canvas, (7, (i * 35) + 5),(w, (i * 35) + 35), (0, 0, 255), -1)
		cv2.putText(canvas, text, (10, (i * 35) + 23),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(255, 255, 255), 2)
		cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
		cont = cont + 1

	if cont == 0:
		messagebox.showinfo("Advertencia ..!","La imagen seleccionada no corresponde a una persona")
		return 1
	cv2.imshow('Deteccion Facial', frameClone)	

def detectarFotografia():
	ruta = askopenfilename()
	imagen = cv2.imread(ruta)
	while True:
		bandera = procesarDatos(imagen)
		if cv2.waitKey(1) & 0xFF == ord('q') or bandera == 1:
			break

	cv2.destroyAllWindows()

def detectarVideo():
	camera = cv2.VideoCapture(0)
	while True:
		frame = camera.read()[1]
		procesarDatos(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	camera.release()
	cv2.destroyAllWindows()


ventana = Tk()
ventana.title("PROCESAMIENTO DE IMAGENES")
ventana.geometry("800x500")
ventana.configure(background="blue")
fondo=PhotoImage(file="Imagenes/ola.png")
fondo2=PhotoImage(file="Imagenes/imagen.png")
fondo3=PhotoImage(file="Imagenes/camara.png")
lblFondo=Label(ventana,image=fondo).place(x=0,y=0) #fondo

etiqueta1 = Label(ventana,text="SISTEMA DE DETECCIÓN DE EMOCIONES", background="slategray",font=("Helvetica", 16, "bold")).place(x=20,y=20)

etiqueta2 = Label(ventana,text="Detección de emociones Faciales en fotografías", background="slategray",font=("Helvetica", 10, "bold")).place(x=75,y=70)
lblFondo2 = Label(ventana,image=fondo2).place(x=190,y=100) #fondoImagen
boton1 = Button(ventana,text="Seleccionar la Imagen",command=detectarFotografia, background="silver",font=("Helvetica", 9, "bold")).place(x=160,y=180)

etiqueta3 = Label(ventana,text="Detección de emociones Faciales en tiempo real mediante la WEBCAM", background="slategray",font=("Helvetica", 10, "bold")).place(x=30,y=240)
lblFondo3 = Label(ventana,image=fondo3).place(x=190,y=270) #fondoImagen
boton1 = Button(ventana,text="Iniciar la WEBCAM",command=detectarVideo, background="silver",font=("Helvetica", 9, "bold")).place(x=165,y=350)


ventana.mainloop()