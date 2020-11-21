import numpy as np
import math
from scipy.spatial import distance
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error

class FEMaRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, z=3):
        self.z = z
    
    def fit(self, X, y):
        self._train_count = len(X)
        self._X_train = X
        self._y_train = y
        return self
    
    def probability(self, xk, sum_dist_xk_all_train):
        q = np.array([y_tr for y_tr in self._y_train])# change
        b = np.array([self.shepard(xk, x_tr, sum_dist_xk_all_train) for x_tr in self._X_train])
        return np.dot(q, b)
    
    def sample_prob(self, xk):
        sum_dist_xk_all_train = np.sum([self.weight(xk, x_tr) for x_tr in self._X_train])
        return self.probability(xk, sum_dist_xk_all_train)
    
    def predict(self, X):
        check_is_fitted(self, '_train_count')
        test_count = len(X)
        prob = np.zeros(test_count)
        
        for k in range(test_count):
            sum_dist_xk_all_train = np.sum([self.weight(X[k], x_tr) for x_tr in self._X_train])
            prob[k] += self.probability(X[k], sum_dist_xk_all_train)
        return prob
    
    def weight(self, xk, xj):
        # inverse_distance_weighting
        dist = distance.euclidean(xk, xj) ** self.z
        return (1.0 / dist) if dist != 0.0 else 1.0
    
    def shepard(self, xk, xi, sum_dist_xk_all_train):
        return self.weight(xk, xi) / sum_dist_xk_all_train
    
    def score(self, X, y):
        self.fit(X, y)
        return math.sqrt(mean_squared_error(y, self.predict(X)))

    def get_params(self, deep=True):
        return {'z': self.z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    