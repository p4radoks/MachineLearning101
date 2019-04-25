import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm

dt=pd.read_csv("https://raw.githubusercontent.com/HakkiKaanSimsek/Makine_Ogrenmesi_Dersleri/master/3.karar_agaclari/ml_3a_regresyon/data/housing.csv")

X=dt.drop(["median_house_value"], axis=1)        #Drop ile belirteceğimiz satır veya sütunu düşürüyoruz. axis=1 bize sütunu düşüreceğimizi gösterir.
y=dt["median_house_value"]                       #Modeldeki hedefiimiz ortalama ev değeri olduğu için onu ayırıyoruz.

X=pd.concat([X, pd.get_dummies(X.ocean_proximity)], axis=1)     #pd.concat birleştirme yapar. pd.get_dummies ise makine öğrenmesi algoritması için object değerleri 0 ve 1'lere ayırır. X ile dummies sonrası ayrılmış verileri birleştiriyoruz burada.
X=X.drop(["ocean_proximity"], axis=1)                           #Yukarıda okyanusa yakınlığı 0 ve 1'lere çevirdikten sonra orijinal değerimizi X'den atıyoruz.
X["total_bedrooms"]=X["total_bedrooms"].fillna(X["total_bedrooms"].median())    #Burada evlerdeki yatak odası sayılarında boş kalmaması için boş değerleri ortalama değerle dolduruyoruz.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42) #Veriyi test ve eğitim olarak ayırdık. random_state bize hep aynı test ve eğitim verisini döndürerek aynı çıktıyı almamızı sağlar.
model= DecisionTreeRegressor(max_depth=3, random_state=42)                                 #Karar ağacı modeli oluşturuk.
model.fit(X_train, y_train)                                                                #Test verilerimizi modelimize oturttuk.

predict=model.predict(X_test)                                                  #predict ile X_test verilerini oluşturduğumuz modele sokuyoruz.
errors=abs(predict-y_test)                                                     #Tahmin edilen ile y_test verisi arasındaki farkımız ise bizim hatalarımız.
print("Mean Absolute Error: ", round(np.mean(errors), 2), "unit")              #Burada ise MAE'yi bulmak için bütün hatalarımızın ortalamasını alıp, 2 ondalıklı şekilde yuvarlıyoruz. Bu da bize ortalama hatayı verir.
mape=100*(errors/y_test)                                                       #Burası bize her değerin MAPE değerini verir.
accuracy = 100-np.mean(mape)                                                   #Doğruluk 100-MAPE'lerin ortalamasıdır.
print("Accuracy: ", round(accuracy, 3), "%")


model2=DecisionTreeRegressor(max_depth=12, random_state=42)
model2.fit(X_train, y_train)
predict=model2.predict(X_test)

errors=abs(predict-y_test)
print("Mean Absolute Error: ", round(np.mean(errors), 2), "unit")
mape=100*(errors/y_test)
accuracy = 100-np.mean(mape)
print("Accuracy: ", round(accuracy, 3), "%")

plt.figure(figsize=(16, 9))

ranking=model2.feature_importances_                                             #Özniteliklerimizin önemini göstermek için kkullanılıyor feature_importances_
features=np.argsort(ranking)[::-1][:10]                                         #Burada bizim üstte hesapladığımız önem sıralamsını argsort ile küçükten büyüğe olacak şekilde indexlerini alır. Bunları ters çevirir. Yani büyükten küçüğe görünmesini sağlar ve sadece 10 tanesini gösterir.
columns=X.columns

plt.title("Özniteliklerin Karar Ağacındaki Önemleri")
plt.bar(range(len(features)), ranking[features], align="center")
plt.xticks(range(len(features)), columns[features]   ,rotation=80)
plt.show()





# accuracy = model.score(X_test, y_test)
# print(accuracy)



