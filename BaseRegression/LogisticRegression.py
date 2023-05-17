import numpy as np
import BaseRegression

class LogisticRegression:
    def approximation(self, X, w, b):
        linear_model = np.dot(X, w) + b
        return self._sigmoid(linear_model)
            

    def _predict(self, X, w, b):
        linear_model = np.dot(X, w) + b
        y_hat = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_hat]
        return y_predicted_class

    def _sigmoid(self, X):
        return 1/(1 + np.exp(-X))
    