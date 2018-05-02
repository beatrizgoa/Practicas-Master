__author__ = 'Bea'

import cPickle
import cv2
import dataset
from hog import HOG
import argparse


def main ():

    #-model=./modelo -image=./imagen.jpg

    #Para pasarle los argumentos
    argumento=argparse.ArgumentParser()

    argumento.add_argument('-model', '--model', required = True,
     help = "path where the training model is saved")
    argumento.add_argument("-image", "--image", required = True,
     help = "path where the image, which is going to be clasified, is saved")


    #Se le asigna el primer argumento a la ruta del modelo de entrenamiento
    args = vars(argumento.parse_args())
    model_path = args['model']

    #Se le asigna el segundo argumento ala ruta de la imagen que se va a clasificar
    image_path = args['image']


    #Se carga el entrenamiento
    model=cPickle.load(file(model_path))

    #Se carga la imagen
    image = cv2.imread(image_path)

    #escalo la imagen porque me sale muy grande
    h,w=image.shape[:2]
    image=cv2.resize(image, (w/4,h/4),interpolation=cv2.INTER_AREA)

    #A escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Filtro gausiano
    blur = cv2.GaussianBlur(gray_image,(5,5),0)

    #Bordes canny
    edges = cv2.Canny(blur,150,100)

    #Extraer contornos
    ima, ctrs, hier = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)


    #Obtenemos los recangulos de cada contorno
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    #Definimos los parametros de la Transformada de hog
    hog = HOG(orientations = 18, pixelsPerCell = (10, 10), cellsPerBlock = (1, 1), normalize = True)

    for rect in rects:
        #Draw the rectangles
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

        #Se recorta la region
        crop_region = gray_image[pt1:pt1+leng, pt2:pt2+leng]

        #Umbralizado de otsu e invertimos la imaagen
        ret,threshold = cv2.threshold(crop_region,177,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

        threshold = dataset.deskew(threshold, 20)
        threshold = dataset.center_extent(threshold, (20, 20))

        #Se obtiene el digito a partir del modelo de gradiente orientado y del modelo de entramiento
        hist = hog.describe(threshold)
        digit = model.predict(hist)[0]


        #Se dibuja el digito correspondiente
        cv2.putText(image,str(int(digit)),(rect[0],rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)

    cv2.imshow('ventana',image)
    cv2.waitKey()


if __name__ == '__main__':
    main()



