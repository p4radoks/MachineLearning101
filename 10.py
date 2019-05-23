import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from collections import Counter
import pandas as pd
import random

def model(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("adasdasd")
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt((features[0] - predict[0])**2 + (features[1] - predict[1])**2)  Bu 2 öznitelik için yeterli ama daha fazlası için yeterli değil.
            euclidean_distance = np.linalg.norm((np.array(features))-(np.array(predict)))           #Bütün tahmin değerlerimiz ile diğer özniteliklerimiz arasındaki öklid uzaklığını bulur.
            distances.append([euclidean_distance, group])                                           #öklid uzaklığı ile datamız içerisindeki verileri birleştirir.

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1)[0][0])
    vote_result=Counter(votes).most_common(1)[0][0]

    return vote_result

data = pd.read_csv("breast-cancer-wisconsin .data")
data.replace("?", -99999, inplace=True)
data.drop(["id"], 1, inplace=True)
full_data = data.astype(float).values.tolist()                  #Datamızı bir listeye çeviriyor ve değerleri float a çeviriyor.
random.shuffle(full_data)                                       #Verimizi ramdom karıştırır.

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]         #Eğitim verisinin baştan yüzde 80 olduğunu belirttik.
test_data = full_data[-int(test_size*len(full_data)):]          #Test verisinin son yüzde 20 olduğunu belirttik.
print(train_data)

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = model(train_set, data, k=3)
        if group == vote:
            correct += 1
        total += 1

print("Accuracy: ", correct/total)