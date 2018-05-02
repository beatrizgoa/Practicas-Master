__author__ = 'Bea'

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import cross_val_score
import numpy as np
from Hist2DClassifier import *
import matplotlib.pyplot as plt


class Hist2DClassifier_simple(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.clasificador = None
        self.entrenador = None



    def fit(self, X, y, fit_prior=True, class_prior=None):
        y=np.ravel(y)
        print np.max(y)
        print np.min(y)
        tam_ventana = range(1, 40)
        ventana_scores = []
        for k in tam_ventana:
            histogram = Hist2DClassifier(tam_ventana=k)
            scores = cross_val_score(histogram, X, y, cv=5, scoring='accuracy')
            ventana_scores.append(scores.mean())

        optim_ventana = ventana_scores.index(max(ventana_scores))
        print optim_ventana


        #for k in range(0,n_classes):
        Histograma = Hist2DClassifier(tam_ventana=(optim_ventana + 1))
        entrenamiento = Histograma.fit(X, y)

        self.entrenador = entrenamiento
        self.clasificador = Histograma

        plt.plot(ventana_scores)
        plt.ylabel('scores')
        plt.xlabel('window size values')
        plt.show()

        return self

    def predict(self,X):
        histograma = self.clasificador
        preciccion = histograma.predict(X)	#Predict the class labels for the provided data

        return preciccion