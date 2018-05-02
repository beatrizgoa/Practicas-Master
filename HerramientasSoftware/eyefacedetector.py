import numpy as np
import cv2
import argparse

#--video=./Avengers.mp4 --out=./avengers_result.avi

#Definimos una funcion que realice el reconocimiento de caras y ojos
def Cascada(frame):
 #Ejecutamos detector caras con el clasificador cascasa
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rectsF = faceCascade.detectMultiScale(frame,
    scaleFactor = 1.1,
    minNeighbors =5,
    minSize=(30,30),
    flags = cv2.CASCADE_SCALE_IMAGE)

    #Ejecutamos el clasificador cascada, con el entrenador en la ruta correspondiente para reconocer ojos
    EyeCascade = cv2.CascadeClassifier('C:\Users\Bea\Desktop\master\herramientas software\python_workspace\practica2\eye_face'+'\\'+'haarcascade_eye.xml')
    rectsE = EyeCascade.detectMultiScale(frame,
    scaleFactor = 3,
    minNeighbors =1,
    minSize=(8,8),
    flags = cv2.CASCADE_SCALE_IMAGE)

    #Hacemos los rectangulos a los ojos detectados y se guarda en frame
    for (x, y, w, h) in rectsE:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) #Los ojos se muestran en verde

    #Hacemos los rectangulos a las caras detectadas y atmbien se guarda en frame
    for (x, y, w, h) in rectsF:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 100), 2) #La cara se muestra en rojo

    return frame

def main ():
    argumento=argparse.ArgumentParser()

    argumento.add_argument('-v', '--video', required = True,
     help = "path to where the video file resides")
    argumento.add_argument("-out", "--out", required = True,
     help = "path to where the video will be saved")

    #Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['video']

    #Se le asigna el segundo argumento ala ruta de salida
    out_path = args['out']



    #Se prepara el video para leer y para grab
    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
    out = cv2.VideoWriter(out_path,fourcc, 28.0, (640,266)) #resolucion igual que la del video de entrada

    #Mientras que se lee el video...
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret==True:
            #Se llama a la funcion cascada que es la que detecta y dibuja ojos y caras
            captura = Cascada(frame)
            #captura = Cascada(frame)
            #Muestre por pantalla y guarde los frames con los rectangulos
            cv2.imshow('eye and face detection',captura)
            out.write(captura)

            #Cuando se presione la letra 'q' en el teclado, que se salga
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()