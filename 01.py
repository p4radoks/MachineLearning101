import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('https://raw.githubusercontent.com/HakkiKaanSimsek/Makine_Ogrenmesi_Dersleri/master/1.aciklayici_veri_analizi/data/adult.csv') #pandas kütüphanesini kullanarak belirtilen linkten verimizi çektik.
df.columns=["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationships", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"]       #sütunlarımıza isimlerini verdik. Sıralama önemli.

df.info()   #Bu bize verilerimizin miktarını, null değer olup olmadığını ve verilerin türünü gösterir. Ayrıca kullandığı bellek alanını da.

plt.figure(figsize=(8,6))                   #Grafiğimizin genişliğini ve yüksekliğini belirtmemizi sağlar. 8 genişlik, 6 ise yüksekliktir.
sns.countplot(df["salary"])                 #seaborn kütüphanesi bize verimizi görselleştirir. countplot verimizin içerisindeki numeric olmayan gözlemi kategorik olarak gösterir.
print("Salary Distribution: ")              #Salary Distribution: Maaş Dağılımı
print(df["salary"].value_counts())          #Ücret içerisindeki değerlerin sayısını gösterir. Yani 50k altı ve 50k üstü kaç tane maaş var onların değerlerini verir.


print("Maaşı 50k altı olanların yaş karakteristiği: ")
print(df[df["salary"]== " <=50K"].age.describe())                   #Maaşı 50k altı olanları yaş ile ilişkili halini gösterir. Mesele mean'de 50k altı maaş alan kişilerin yaş ortalamasını gösterir.
print("")
print("Maaşı 50K'dan yüksek olanların yaş karakteristiği:")
print(df[df['salary'] == " >50K"].age.describe())

plt.figure(figsize=(12,8))
sns.countplot(df["education"], order=df["education"].value_counts().index)  #Burada verileri sayılarına göre grafikte gösterdik.
plt.xticks(rotation=70)                                                     #Grafiğin x ekseninindeki yazıları 70 derece döndürür.
print(df["education"].value_counts())

print(df.groupby("education")["age"].describe())                            #Eğitim düzeyi ile yaş arasındaki ilişkiyi gösterir. 10 sınıfa kadar okuyanların ortalama yaşı, min yaşı gibi verileri verir.

plt.figure(figsize=(12,8))
sns.barplot(x="education", y="hours-per-week", data=df, hue="salary", palette="inferno")    #Burada eğitim ile haftalık çalışma saatinin arasındaki ilişki gösteriliyor. Ayrıca maaşa göre de ayrımı mevcut.
plt.xticks(rotation=70)

plt.figure(figsize=(12,8))
sns.barplot(x="education", y="hours-per-week", data=df, hue="sex", palette="dark")
plt.xticks(rotation=70)

f, (ax1, ax2) = plt.subplots(1,2, figsize=(16,7))                                                                                    #Birden fazla grafiği aynı pencerede gösterme.
sns.pointplot(x="education", y="hours-per-week", hue="sex", palette="inferno", data=df[df["salary"] == " <=50K"], ax=ax1)
ax1.set_title("Salary = <=50K") #1. grafiğin başlığı
sns.pointplot(x="education", y="hours-per-week", hue="sex", palette="dark", data=df[df["salary"] == " >50K"], ax=ax2)
ax2.set_title("Salary = >50K")  #2.grafiğin başlığı
f.autofmt_xdate(rotation=50)

plt.show()                                   #Kendisinden sonra yazdırılanları yok sayıyor. En sona koymak lazım bu satırı.