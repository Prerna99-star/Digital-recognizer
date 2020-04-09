# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:14:40 2020

recognization of handwritting
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data= pd.read_csv('train.csv').as_matrix()
clf = DecisionTreeClassifier()

xtrain = data[0:21000, 1:]
train_label = data[0:21000, 0]

clf.fit(xtrain, train_label)

xtest = data[21000: , 1:]
actual_label = data[21000: , 0]

#predicting the test data
d = xtest[8]
d.shape = (28,28)
plt.imshow(255-d, cmap = 'gray')
plt.show()
print(clf.predict())

# getting the accuracy
p = clf.predict(xtest)

count = 0
for i in range(0, 21000):
    count+=1 if p[i] == actual_label[i] else 0
    print("Accuracy = ", (count/21000)*100)