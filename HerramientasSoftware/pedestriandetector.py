__author__ = 'Bea'
#################
# Ejercicio 3 poco a poco
##################

#impoorts
import cv2
from PIL import Image
import glob, os
import numpy as np
import argparse


def main ():
    argumento=argparse.ArgumentParser()

    argumento.add_argument('-images', '--images', required = True,
     help = "path to where the pedestrian files reside")
    argumento.add_argument("-out", "--out", required = True,
     help = "path to where the video will be saved")

    #Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    in_path = args['images']


    #Se le asigna el segundo argumento ala ruta de salida
    out_path = args['out']



    imagenes = os.listdir(in_path)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')
    out = cv2.VideoWriter(out_path,fourcc, 38.0, (640,480))

    #Leemos cada una de las imagenes
    for posicion, imagen in enumerate(imagenes):
        im= Image.open(in_path + '\\' + imagen)
        im=np.array(im)

        #Detectamos a los peatones
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector())
        hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}
        (result,weights) = hog.detectMultiScale(im, **hogParams)


        #hacemos el rectangulo al redeodr  de los peatones
        for (x,y,w,h) in result:
            cv2.rectangle(im,(x, y), (x + w, y + h), (0, 0, 255), 2)

        #mostramos
        cv2.imshow('pedestrian detection',im)
        cv2.waitKey(20)

        #Cuando se presione la letra 'q' en el teclado, que se salga
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(im)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()