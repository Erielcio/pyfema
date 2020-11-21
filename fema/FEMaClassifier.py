import numpy as np
import math
from scipy.spatial import distance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

class FEMaClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, z=3):
        self.z = z
    
    def fit(self, X, y):
        self._train_count = len(X)
        self._X_train = X
        self._y_train = y
        self._class_count = len(np.unique(y))
        return self
    
    def class_probability(self, xk, c, sum_dist_xk_all_train):
        q = np.array([1 if y_tr == c else 0 for y_tr in self._y_train])
        b = np.array([self.shepard(xk, x_tr, sum_dist_xk_all_train) for x_tr in self._X_train])
        return np.dot(q, b)
    
    def certainty(self, xk, class_index):
        sum_dist_xk_all_train = np.sum([self.weight(xk, x_tr) for x_tr in self._X_train])
        return self.class_probability(xk, class_index, sum_dist_xk_all_train)
    
    def predict_prob(self, X):
        test_count = len(X)
        prob = np.zeros((test_count, self._class_count))
        
        for k in range(test_count):
            sum_dist_xk_all_train = np.sum([self.weight(X[k], x_tr) for x_tr in self._X_train])
        
            for c in range(self._class_count):
                prob[k, c] += self.class_probability(X[k], c, sum_dist_xk_all_train)
        return prob
    
    def predict(self, X):
        check_is_fitted(self, '_train_count')
        prob = self.predict_prob(X)
        return np.array([np.argmax(p) for p in prob])
    
    def weight(self, xk, xj):
        # inverse_distance_weighting
        dist = distance.euclidean(xk, xj) ** self.z
        return (1.0 / dist) if dist != 0.0 else 1.0
    
    def shepard(self, xk, xi, sum_dist_xk_all_train):
        return self.weight(xk, xi) / sum_dist_xk_all_train
    
    def score(self, X, y):
        self.fit(X, y)
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {'z': self.z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    