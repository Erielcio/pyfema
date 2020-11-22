import numpy as np
import math
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

class FEMaClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, z=3):
        self._z = z
    
    def fit(self, X, y):
        self._train_count = len(X)
        self._class_count = len(np.unique(y))
        self._X_train = X
        self._y_train = y
        return self
    
    def predict_prob(self, X):
        check_is_fitted(self, '_train_count')
        test_count = len(X)
        prob_per_class = np.zeros((test_count, self._class_count))
        for k in range(test_count):
            phi = self.shepard(X[k], self._X_train)
            for c in range(self._class_count):
                prob_per_class[k, c] = np.dot(np.fromiter((1 if y_tr == c else 0 for y_tr in self._y_train), dtype=float), phi)                
        return prob_per_class
    
    def predict(self, X):
        check_is_fitted(self, '_train_count')
        return np.argmax(self.predict_prob(X), axis=1)
    
    def shepard(self, xk, X_train):
        diff = xk - X_train
        diff = np.where(diff == 0, 0.000001, diff) 
        dist = np.linalg.norm(diff, axis=1)
        idw = 1.0 / np.power(dist, self._z)
        return idw / np.sum(idw)
    
    def score(self, X, y):
        self.fit(X, y)
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {'z': self._z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self