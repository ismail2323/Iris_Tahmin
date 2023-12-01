# Iris_Tahmin
MakineOgrenimi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X = iris.data 
y = iris.target 
feature_names = iris.feature_names 
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df = pd.DataFrame(X_train, columns=feature_names)
df['target'] = y_train
df.describe()

sns.pairplot(df, hue='target', vars=feature_names)
plt.show()

params = {'n_neighbors': range(1, 11)}
grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
grid.fit(X_train, y_train)
print("En iyi k parametresi:", grid.best_params_)

knn = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Accuracy score:", accuracy_score(y_test, y_pred))

