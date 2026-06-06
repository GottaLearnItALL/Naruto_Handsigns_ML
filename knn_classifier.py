from math import sqrt
import numpy as np
from collections import Counter

class KNNCLassifier():
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X_train, y_train):
        """ Just store the training data KNN needs no training """
        self.X_train = X_train
        self.y_train = y_train
    
    def _euclidean_distance(self, a, b):
        distance = np.sqrt(np.sum((a-b)**2))
        return distance
    
    def _manhattan_distance(self, a, b):
        distance = np.sum(np.abs(a-b))
        return distance
    
    def _predict_single(self,x):
        """ Predict the class for a single data point x """
        distance_from_x = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_nearest = np.argsort(distance_from_x)[:self.k]
        k_labels = [self.y_train[i] for i in k_nearest]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X_test):
        """Predict classes for an array of data points."""
        return np.array([self._predict_single(x) for x in X_test])


X_train = np.array([[1,2], [2,3], [3,1], [6,5], [7,8]])
y_train = np.array([0, 0, 0, 1, 1])
x_test = np.array([[2,2]])  # should predict 0 (closer to the first cluster)

knn = KNNCLassifier(k=3)
knn.fit(X_train, y_train)
prediction = knn._predict_single(x_test[0])
print(f"Prediction: {prediction}")