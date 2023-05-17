import numpy as np

class BaseRegression:
    def __init__(self, n_iters, learning_rate):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_hat = np.dot(X, self.weights) + self.bias

            dw = (1/ n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return self._predict(X, self.weights, self.bias)

    def approximation(self, X, w, b):
        raise NotImplementedError()
    
    def _predict(self, X, w, b):
        raise NotImplementedError()
