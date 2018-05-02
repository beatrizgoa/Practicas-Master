__author__ = 'Beatriz Gomez Ayllon'

from sklearn import neighbors
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.cross_validation import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
class KnnClassifier(BaseEstimator, ClassifierMixin):


    def __init__(self, class_prior = None):
        self.entrenador=None
        self.clasificador=None
        self.mean_ = None
        self.cov_ = None
        self.invcov_ = None

    def fit(self, X, y):

        '''
        parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        fit_prior: boolean
            If True the priors are estimated from data. When False
            the priors are in class_prior parameter
        class_prior: array, shape (n_classes, 1)
             Priors values for each class
        '''
        y=np.ravel(y)

        k_range = range(1, 30)
        k_scores = []
        for k in k_range:
            neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(neigh, X, y, cv=5, scoring='accuracy')
            k_scores.append(scores.mean())

        optim_k = k_scores.index(max(k_scores))
        print optim_k

        #for k in range(0,n_classes):
        neigh = neighbors.KNeighborsClassifier(n_neighbors=(optim_k + 1))

        entrenamiento = neigh.fit(X, y)

        self.entrenador = entrenamiento
        self.clasificador = neigh

        plt.plot(k_scores)
        plt.ylabel('scores')
        plt.xlabel('k values')
        plt.show()



    def predict(self,X):
        neigh = self.clasificador
        preciccion = neigh.predict(X)	#Predict the class labels for the provided data

        return preciccion

