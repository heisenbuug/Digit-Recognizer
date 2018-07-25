import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv").as_matrix()

dt_clf = DecisionTreeClassifier()
# training dataset
xtrain = data[0: 21000, 1:]
train_label = data[0:21000, 0]

dt_clf.fit(xtrain, train_label)

# testing data
xtest = data[21000: , 1:]
test_label = data[21000: , 0]

p = dt_clf.predict(xtest)

# To Display any number, here i am printing xtest[8]
d = xtest[8]
d.shape = (28, 28)
pt.imshow(255 - d, 'gray')
print(dt_clf.predict([xtest[8]]))
pt.show()

# To check Accuracy
count = 0

for i in range(0, 21000):
    count+=1 if p[i] == test_label[i] else 0
    
print("Accuracy: ", (count/21000) * 100)
