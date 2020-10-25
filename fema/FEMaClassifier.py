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
    
    def predict(self, X):
        check_is_fitted(self, '_train_count')
        
        test_count = len(X)
        prob = np.zeros((test_count, self._class_count))
        y_pred = []
        
        def weight(xk, xj):
            # inverse_distance_weighting
            dist = math.pow(distance.euclidean(xk, xj), self.z)
            return (1.0 / dist) if dist != 0.0 else 1.0
    
        def shepard(xk, xi, sum_dist_xk_all_train):
            return weight(xk, xi) / sum_dist_xk_all_train
        
        for k in range(test_count):
            sum_dist_xk_all_train = 0.0
                                
            for j in range(self._train_count):
                sum_dist_xk_all_train += weight(X[k], self._X_train[j])
            
            for c in range(self._class_count):
                for i in range(self._train_count):
                    prob[k, c] += (1 if self._y_train[i] == c else 0) * shepard(X[k], self._X_train[i], sum_dist_xk_all_train)
        
        for k in range(test_count):
            y_pred.append(np.argmax(prob[k]))
        
        return y_pred
    
    def certainty(self, X, class_index):
        # coming soon...
        return -1
    
    def score(self, X, y):
        self.fit(X, y)
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {'z': self.z}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    