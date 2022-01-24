import numpy as np
import pandas as pd

dataset = pd.read_csv(‘mushrooms.csv’)

X = dataset.iloc[:, 1:23].values
y = dataset.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
for i in range(0, 22):
 X[:, i] = labelencoder_X.fit_transform(X[:, i])

for i in range(0, 22):
 columnTransformer = ColumnTransformer([(‘encoder’, OneHotEncoder(), [0])],
remainder=’passthrough’)
 X=np.array(columntransformer.fit_transform(X), dtype=str)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle= True, random_state = 1)

from sklearn.neighbors import KNeighborsClassifir
classifier = KNeighborsClassifier(n_neighbors = 10, metric = ‘minskowski’, p=2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuraries.mean()
accuraries.std()

