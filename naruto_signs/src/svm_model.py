from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import joblib

param_grid = {
    'C': [0.1,1,10,100],
    'kernel': ['rbf','linear']
}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5, scoring='accuracy')


data = np.load("naruto_signs/data/features_train.npz")
features = data['features']
labels = data['labels']

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)



grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV accuracy: {grid.best_score_*100:.1f}%")

best_svm = grid.best_estimator_
accuracy = np.sum(best_svm.predict(X_test) == y_test) /len(y_test)
print(f"Test accuracy: {accuracy*100:.1f}%")


joblib.dump(best_svm, "naruto_signs/models/svm_model.pk1")
