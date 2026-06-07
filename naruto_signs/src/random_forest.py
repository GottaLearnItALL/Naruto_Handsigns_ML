import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = np.load("naruto_signs/data/features_train.npz")
features = data['features']
labels =  data['labels']


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
prediction = rf.predict(X_test)

accuracy = np.sum(prediction == y_test) / len(y_test)

print(f"Random Forest Accuracy: {accuracy*100:.1f}%")