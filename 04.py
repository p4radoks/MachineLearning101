import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from sklearn import ensemble

df = pd.read_csv('https://raw.githubusercontent.com/HakkiKaanSimsek/Makine_Ogrenmesi_Dersleri/master/3.karar_agaclari/ml_3b_siniflandirma/data/adult.csv')
df.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race", "sex",
              "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]

X = df.drop(["salary"], axis=1)
y = df["salary"]

df["capital-gain"] = df["capital-gain"].astype(float)  # capital-gainin değerlerini floata çevirdik.
# print(X.select_dtypes(include="object").tail(20))

categorical_columns = [c for c in X.columns if
                       X[c].dtype.name == "object"]  # Data tipi object olanları değişkene atadık.
for c in categorical_columns:  # Atadığımız değişkendekileri for döngüsü ile cevabı ? işareti olanların hepsini mod değeriyle doldurduk. df'i de aynı şekilde.
    X[c] = np.where(X[c] == " ?", X[c].mode(), df[c])
# print(X.select_dtypes(include="object").tail(20))

X = pd.concat([X, pd.get_dummies(X.select_dtypes(include="object"))], axis=1)
X = X.drop(["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

prediction = model.predict(X_test)
score = round(accuracy_score(y_test, prediction), 3)
cm1 = cm(y_test, prediction)
print(cm1)
print(score)
print(classification_report(y_test, prediction, target_names=[" <=50K", " >50K"]))

plt.figure(figsize=(16, 9))
ranking = model.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns

plt.title("Özniteliklerin Önem Sıralaması")
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()
