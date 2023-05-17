import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from LinearRegression_class import LinearRegression
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=1234)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
predicted = linear_model.predict(X_test)

def mse(true, predicted):
    return np.mean((true - predicted) **2)

mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line = linear_model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9))
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()