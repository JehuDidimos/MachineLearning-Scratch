# %%
from LinearRegression_class import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv("TSLA.csv")
X= dataset[["High", "Open", "Low", "Volume"]]
y= dataset["Adj Close"]


# %%
def mse(true, prediction):
    return np.mean((true - prediction) ** 2)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)
print(y_test)

# %%
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
prediction = linear_model.predict(X_test)
print(prediction)
print(mse(y_test, prediction))


