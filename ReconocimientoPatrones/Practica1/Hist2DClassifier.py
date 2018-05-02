__author__ = 'Bea'

from sklearn import cross_validation,covariance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import cross_val_score
import numpy as np
import math
import matplotlib.pyplot as plt

class Hist2DClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,tam_ventana):
        self.clasificador = None
        self.tam_matriz = None
        self.bordes = [0,0]
        self.num_gausianas = None
        self.tam_ventana = tam_ventana





    def fit(self, X, y, fit_prior=True, class_prior=None):
        num_gausianas = np.amax(y)+1

        xmin = math.ceil(np.amin(X))
        xmax = math.floor(np.amax(X))
        bordes = [xmax,xmin]
        tam_matriz = math.ceil((xmax + abs(xmin))/self.tam_ventana)
        matriz =  np.zeros([num_gausianas,tam_matriz,tam_matriz])
        [tamx,tamy] = X.shape

        #Se calcula la matriz con el numero de muestras por bin
        for pos in range(0,tamx):
            p = X[pos]
            P0 = float(p[0])
            P1 = float(p[1])

            posx = math.trunc((P0+abs(xmin))/self.tam_ventana)
            posy = math.trunc((P1+abs(xmin))/self.tam_ventana)

            pos_xm = posx - 1
            pos_ym = posy - 1
            clase = y[pos]

            matriz[clase][pos_xm][pos_ym]=matriz[clase][pos_xm][pos_ym]+1

        self.clasificador = matriz
        self.tam_matriz = tam_matriz
        self.bordes = bordes
        self.num_gausianas = num_gausianas

        return self


    def predict(self, X):
        y=np.zeros(len(X))
        matriz =self.clasificador
        num_gausianas = int(self.num_gausianas)
        acumulaciones=np.zeros(num_gausianas)
        [tamx,tamy] = X.shape
        tam_matriz = self.tam_matriz
        [xmax,xmin] = self.bordes

        #Se calcula la matriz con el numero de muestras por bin
        for i in range(0,tamx):
            p0 = X[i][0]
            p1 = X[i][1]

            #Si la muestra tiene un valor x o y mayor o menor que los bordes de  la matriz se le asignan los bordes
            # y se clasifica en funcion a ese nuevo valor

            if p0<xmin:
                p0=xmin+2

            if p1<xmin:
                p1=xmin+2

            if p0>xmax:
                p0=xmax-2

            if p1>xmax:
                p1=xmax-2

            for j in range(0,num_gausianas):
                posx= math.trunc((p0+abs(xmin))/self.tam_ventana)
                posy= math.trunc((p1+abs(xmin))/self.tam_ventana)

                pos_xm = posx - 1
                pos_ym = posy - 1
                acumulaciones[j]=matriz[j][pos_xm][pos_ym]

            etiqueta = np.argmax(acumulaciones)
            y[i] = etiqueta

        return y




