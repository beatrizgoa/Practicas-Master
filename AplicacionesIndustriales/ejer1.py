__author__ = 'Bea'

import cv2
import matplotlib
import math
import numpy as np
import os
from PIL import Image
import numpy.matlib


ruta_directorio='C:\Users\Bea\Desktop\master'+'\\'+'aplicaciones industriales\Practica1\Muestra'

imagenes = os.listdir(ruta_directorio)

#Leemos cada una de las imagenesh
for posicion, imagen in enumerate(imagenes):
    im= cv2.imread(ruta_directorio + '\\' + imagen)
    image=np.array(im)

    [rows,cols,chanel]=image.shape

    #A escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Filtro gausiano para eliminar ruido
    blur = cv2.GaussianBlur(gray_image,(5,5),0)

    a,b=blur.shape
    im_thres=np.matlib.zeros((a, b))

    #Se hace la umbralizacion
    thr1=30
    thr2=85

    for nfila, vector in enumerate(blur):
       for ncolumna, pixel in enumerate(vector):
          if pixel>thr1 and pixel<thr2:
            blur[nfila][ncolumna] = 255
          else:
             blur[nfila][ncolumna] = 0


    #ret,im_thres = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
    #ret2,im_thres2 = cv2.threshold(im_thres,30,255,cv2.THRESH_BINARY_INV)



    #Operaciones morfologicas
    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, (cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))))

    im_ero = cv2.erode(opening,(cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))),iterations =2)
    im_dil = cv2.dilate(im_ero,cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)),iterations =3)


    #Bordes con canny
    edges = cv2.Canny(im_dil,cv2.HOUGH_GRADIENT,0,255)

    # Se inicializan variables
    theta_h = np.pi/180
    rho_h = 1
    y_h = 0
    angulo_horizontal = 0

    lines = cv2.HoughLines(edges,1,np.pi/180,70)


    for i in range (len(lines)):
        for rho_h,theta_h in lines[i]:
            a = np.cos(theta_h)
            b = np.sin(theta_h)
            x0 = a*rho_h
            y0 = b*rho_h
            x1 = int(x0+1000*(-b))
            y1 = int(y0+1000*(a))
            x2 = int(x0-1000*(-b))
            y2 = int(y0-1000*(a))

            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

            if (((theta_h>math.pi*92/180) and (theta_h<math.pi*190/180) and (y1>y_h))):
                angulo_horizontal = theta_h
                y_h = y1
                x3 = x1
                x4 = x2
                y3 = y1
                y4 = y2

    (M,N) = im_dil.shape
    rotation_matrix = cv2.getRotationMatrix2D((a/2,b/2),angulo_horizontal*(180/np.pi)-90,1.0)
    result = cv2.warpAffine(im_dil, rotation_matrix,(b,a),flags=cv2.INTER_LINEAR)
    edges2 = cv2.Canny(result,20,20, apertureSize= 3)

    ima, ctrs, hier = cv2.findContours(edges, cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
      #Obtenemos los recangulos de cada contorno

    cnts = sorted(ctrs, key = cv2.contourArea, reverse = True)[:3]

    rects = [cv2.boundingRect(ctr) for ctr in cnts]

    for rect in rects:
       #Draw the rectangles
       cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

       # Make the rectangular region around the digit
       leng = int(rect[3] * 1.6)
       pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
       pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

    cv2.imshow("ventana", image)
    cv2.waitKey()

