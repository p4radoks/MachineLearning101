import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter


data= {"k": [[1, 2], [2, 3], [3, 4]], "r": [[6, 8], [7, 5], [8, 9]]}
predict = [3,8]


def model(data,predict, k=3):
    if len(data) >=k:
        warnings.warn("adasdasd")
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = np.sqrt((features[0] - predict[0])**2 + (features[1] - predict[1])**2)  Bu 2 öznitelik için yeterli ama daha fazlası için yeterli değil.
            euclidean_distance = np.linalg.norm((np.array(features))-(np.array(predict)))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1)[0][0])
    vote_result=Counter(votes).most_common(1)[0][0]

    return vote_result

result = model(data,predict, k=3)
print(result)


for i in data:
    for ii in data[i]:
        plt.scatter(ii[0], ii[1], s=100, color=i)
plt.scatter(predict[0], predict[1], color=result)
plt.show()