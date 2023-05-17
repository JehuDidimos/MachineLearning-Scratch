import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from NaaiveBayes import NaiveBayes

def accuracy(true, pred):
    accuracy = np.sum(true == pred) / len(true)
    return accuracy

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
print("Accuracy: ", accuracy(y_test, pred))