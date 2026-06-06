from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter



data = np.load("naruto_signs/data/features_train.npz")
features = data['features']
labels = data['labels']



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

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

knn = KNNCLassifier(k=5)

knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"KNN Accuracy: {accuracy * 100:.1f}%")

