from LinearRegression_class import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv("TSLA.csv")
X = dataset[["High", "Open", "Low", "Volume"]]
y = dataset[["Adj Close"]]

# Scale the input features and the target variable
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train the model
linear_model = LinearRegression(learning_rate=0.65, n_iterations=1000)
linear_model.fit(X_train, y_train)
prediction = linear_model.predict(X_test)

# Calculate the mean squared error
mse = np.mean((sc_y.inverse_transform(y_test) - sc_y.inverse_transform(prediction)) ** 2)
print("Mean squared error:", mse)

# Plot the predictions against the actual values
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
plt.plot(y_test, sc_y.inverse_transform(prediction), 'o', color='purple')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
