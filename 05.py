import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("leaf.csv")
df.columns=["Class", "Specimen_Number", "Eccentricity", "Aspect_Ratio", "Elongation",
         "Solidity", "Stochastic_Convexity", "Isoperimetric_Factor", "Maximal_Indendation_Depth",
         "Lobedness", "Average_Intensity", "Average_Contrast", "Smoothness", "Third_moment", "Uniformity", "Entropy"]


X = df.drop(["Class"], axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9, metric="minkowski")
model.fit(X_train, y_train)

predictions=model.predict(X_test)
accuracy=accuracy_score(y_test, predictions)

print(predictions)
print(accuracy)


