import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class Knn:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        predicted_values = [self.predict_helper(x) for x in X]
        return np.array(predicted_values)

    def predict_helper(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indeces = np.argsort(distances) [:self.k]
        k_nearest = [self.y_train[i] for i in k_indeces]
        most_Common = Counter(k_nearest).most_common(1)
        return most_Common[0][0]


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = Knn(k = 3)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
accuracy = np.sum(prediction == y_test) / len(y_test)
print(accuracy)


