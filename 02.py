import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('https://raw.githubusercontent.com/HakkiKaanSimsek/Makine_Ogrenmesi_Dersleri/master/2.gorsellestirme/data/flight.csv')     #Uçuş bilgilerini içeren veriseti

df["time_hour"]=pd.to_datetime(df["time_hour"])             #Verilerimizi tarihe göre zenginleştirdik. Uçuğun saati, ayı vs gibi değerleri de gösterdik böylece.
df["year"]=df["time_hour"].dt.year
df["month"]=df["time_hour"].dt.month
df["day"]=df["time_hour"].dt.day
df["hour"]=df["time_hour"].dt.hour
df["day_of_week"]=df["time_hour"].dt.dayofweek

#Bundan sonrası veri görselleştirme ile alakalıdır ve https://medium.com/data-science-tr/makine-%C3%B6%C4%9Frenmesi-dersleri-2-e246428de84e
                                                #veya https://veribilimcisi.com/2017/09/06/seaborna-hizli-bir-baslangic/#Box%20Plots adreslerinden bakılabilir.

plt.figure(figsize=(12,9))
sns.boxplot(x="carrier", y="distance", data=df)
plt.xticks(rotation=80)
plt.show()

