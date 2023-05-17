import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender = LabelEncoder()
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
sc_X = StandardScaler()

dataset = pd.read_csv("iphone.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 3].values
print(X)
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

X = np.vstack(X[:,:]).astype(np.float)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ", accuracy)
precision = metrics.precision_score(y_test, y_pred)
print("Precision Score: ", precision)
recall = metrics.recall_score(y_test, y_pred)
print("Recall score: ", recall)



