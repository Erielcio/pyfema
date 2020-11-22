import numpy as np
import math
from scipy.spatial import distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error

class FEMaRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, z=3):
        self._z = z
    
    def fit(self, X, y):
        self._train_count = len(X)
        self._X_train = X
        self._y_train = y
        return self
    
    def predict(self, X):
        check_is_fitted(self, '_train_count')
        return np.fromiter((np.dot(self._y_train, self.shepard(xk, self._X_train)) for xk in X), dtype=float)
    
    def shepard(self, xk, X_train):
        diff = xk - X_train
        diff = np.where(diff == 0, 0.000001, diff) 
        dist = np.linalg.norm(diff, axis=1)
        idw = 1.0 / np.power(dist, self._z)
        return idw / np.sum(idw)
    
    def score(self, X, y):
        self.fit(X, y)
        return math.sqrt(mean_squared_error(y, self.predict(X)))

    def get_params(self, deep=True):
        return {'z': self._z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self