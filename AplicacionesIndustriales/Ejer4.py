import numpy as np
import cv2
import argparse
import theano

def nothing(x):
    pass

def build_filters(ksize, sigma, theta, lam ,gamma,fase):
    filters = []
    kern = cv2.getGaborKernel((ksize,ksize), sigma, theta, lam,gamma, fase, ktype=cv2.CV_32F)
    filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':

 argumento=argparse.ArgumentParser()

 argumento.add_argument('-image', '--image', required = True,
 help = "Ruta donde reside la imagen que se quiere analizar")

 #Se le asigna el primer argumento a la ruta de entrada
 args = vars(argumento.parse_args())
 in_path = args['image']

 #imge = cv2.imread('68.png')
 imge = cv2.imread(in_path)
 img = cv2.cvtColor(imge, cv2.COLOR_BGR2GRAY)

 cv2.namedWindow('image')
 cv2.imshow('imagen_recuadrada',img)

 # create trackbars for color change
 cv2.createTrackbar('ksize','image',80,200,nothing)
 cv2.createTrackbar('sigma','image',2, 120,nothing)
 cv2.createTrackbar('theta','image',19, 120,nothing)
 cv2.createTrackbar('lambda','image',1,50,nothing)
 cv2.createTrackbar('gamma','image',14,100,nothing)
 cv2.createTrackbar('fase','image',0,157,nothing)

 while True:
    # get current positions of four trackbars
    ksize = cv2.getTrackbarPos('ksize','image')
    sigma = cv2.getTrackbarPos('sigma','image')
    theta = cv2.getTrackbarPos('theta','image')
    lam = cv2.getTrackbarPos('lambda','image')
    gamma = cv2.getTrackbarPos('gamma','image')
    fase = cv2.getTrackbarPos('fase','image')

    #se llama a la funcion del filtro
    filters = build_filters(ksize, sigma, theta, lam,gamma,fase)
    res1 = process(img, filters)
    #Se va mostrando por pantalla
    cv2.imshow('Gabor', res1)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        break


    #Se apoya con preprocesado lo obtenido
    ret,im_thres = cv2.threshold(res1,200,255,cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(im_thres, cv2.MORPH_CLOSE, (cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))))
    ret2,im_thres2 = cv2.threshold(opening,200,255,cv2.THRESH_BINARY_INV)
    im_dil = cv2.dilate(im_thres2,cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)),iterations =3)

    #Extraer contornos
    ima, ctrs, hier = cv2.findContours(im_dil, cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
    #Si los cuadrados son menores que un cierto umbral que se dibujen para evitar que se muestre lapantalla entera
    for c in ctrs:
        rect = cv2.boundingRect(c)
        if rect[2] < 150 and rect[3] < 150:
            x,y,w,h = rect
            cv2.rectangle(imge,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('imagen_recuadrada',imge)
    #Si se pulsa q que salga
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


 cv2.destroyAllWindows()