__author__ = 'Bea'

#IMPORTS
import os, glob
from PIL import Image
import numpy as np
import cv2
import argparse




def main ():

    #Definimos variables
    lista_matches=[]
    list_kp=[]

    value=0
    pos=0


    #Para pasarle los argumentos
    argumento=argparse.ArgumentParser()

    argumento.add_argument('-query', '--query', required = True,
     help = "path where the image is saved")
    argumento.add_argument("-covers", "--covers", required = True,
     help = "path where the images which are going to be compared are saved")


    #Se le asigna el primer argumento a la ruta de entrada
    args = vars(argumento.parse_args())
    path_imagen_principal = args['query']

    #Se le asigna el segundo argumento ala ruta de salida
    ruta_directorio = args['covers']

    #Leemos el directorio de las imagenes que se comparan con la principal
    imagenes = os.listdir(ruta_directorio)

    # Initiate ORB detector. Se ha utilizado este detector (que funciona peor) que el detector SIFT ya que este ultimo necesita ser instalado.
    orb = cv2.ORB_create()

    #Selee la iamgen principal
    img1 = cv2.imread(path_imagen_principal)          # queryImage

    #Leemos cada una de las imagenes
    for posicion, imagen in enumerate(imagenes):
        im= Image.open(ruta_directorio + '\\' + imagen)
        img2=np.array(im)


        #find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        list_kp.insert(posicion,kp2) #Lista donde se guardar los KP de las imagenes a comparar

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        #Lista de los matches de todas las imagenes
        lista_matches.insert(posicion,matches)


    #Se lee en la lista de matches el numero de coincidencias y nos quedamos con el de mayor valor y su posicion
    for posicion, matches in enumerate(lista_matches):
        resultados = len(matches)
        if resultados>=value:
            value=resultados
            pos=posicion

    #Si las coincidencias (matches) no superan los 50, no hay matching
    if value <=50:
        print ("NO existe en el directorio una imagen que corresponda con la que buscas")

    else:

        #Nos quedamos con el KP que corresponda con el de la posicion de mayor matching
        kp3=list_kp[pos]
        #Buscaos la imagen que corresponde con ese matching maximos
        img3=np.array(Image.open(ruta_directorio + '\\' + imagenes[pos]))
        #Y los matches
        matches=lista_matches[pos]

        #Calculamos la imagen resultante..
        img_result = cv2.drawMatches(img1,kp1,img3,kp3,matches[:50], None, flags=2)

        #... y se muestra
        cv2.imshow('imagenes coincidentes',img_result)
        cv2.waitKey()

if __name__ == '__main__':
    main()

