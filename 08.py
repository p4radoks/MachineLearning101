import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("breast-cancer-wisconsin .data")
df.replace("?", -9999, inplace=True)
df.drop(["id"], 1, inplace=True)

X = np.array(df.drop(["class"],1))
y = np.array(df["class"])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = neighbors.KNeighborsClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)      #Bu ve aşağıdaki kod bloğu aynı değerleri verir.
print(accuracy)

# prediction = model.predict(X_test)
# a = accuracy_score(y_test, prediction)
# print(a)

examples = np.array([[4,3,2,1,1,3,5,2,7], [4,3,2,1,1,3,5,2,3]])
examples = examples.reshape(2, -1)                                  #Burada 2 satırlık, sütun sayısı belirli olmayan yeni bir matris belirtiyoruz.

prediction = model.predict(examples)
print(prediction)