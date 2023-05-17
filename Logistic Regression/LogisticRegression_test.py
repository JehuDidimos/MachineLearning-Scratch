from LogisticRegression import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

breatcancer_data = datasets.load_breast_cancer()
X, y = breatcancer_data.data, breatcancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

regression_model = LogisticRegression(learning_rate= 0.001, n_iters=1800)
regression_model.fit(X_train, y_train)
prediction = regression_model.predict(X_test)

accuracy = accuracy(y_test, prediction)
print("Accuracy: ", accuracy)
