import random

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd

def k_n_n(data, predict, k=3):
        if len(data) >= k:
                warnings.warn("K values less than total class number!")
        DISTANCE = []
        for group in data:
                for point in data[group]:
                        euc_dist = np.sqrt(np.sum((np.array(point) - np.array(predict))**2))
                        DISTANCE.append([euc_dist, group])

        votes = [i[1] for i in sorted(DISTANCE)[:k]]
        result = Counter(votes).most_common(1)[0][0]

        return result

df = pd.read_csv('Iris.csv')
df.drop('Id', axis=1, inplace=True)
print(df.Species.unique())
data = df.values.tolist()
random.shuffle(data)

test_split = 0.3
train_data = data[:int(test_split*len(data))]
test_data = data[int(test_split*len(data)):]

train = {"Iris-setosa": [], "Iris-versicolor": [], "Iris-virginica": []}
test = {"Iris-setosa": [], "Iris-versicolor": [], "Iris-virginica": []}

for i in train_data:
        train[i[-1]].append(i[:-1])


for i in test_data:
        test[i[-1]].append(i[:-1])

correct = 0
count = 0

for group in test:
        for i in test[group]:
                res = k_n_n(train, i, k=5)
                if res == group:
                        correct+=1
                count +=1

print(f"Acc: {correct/count}")

print(train)
print(test)
#print(df.head())
