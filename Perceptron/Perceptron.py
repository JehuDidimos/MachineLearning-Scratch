import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=100):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.activation_func = self.unit_step_func

    def unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_model)
        return y_pred
    
    def fit(self, X, y):
        n_samples, n_features  = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i >= 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_model)

                delta_update = self.learning_rate * (y[idx] - y_pred)
                self.weights += delta_update * x_i
                self.bias += delta_update