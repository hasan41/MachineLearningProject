import os

import matplotlib as mpl
import matplotlib.pyplot as plt
# from IPython.display import display

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc

Tmatrix = np.loadtxt(open("sign_mnist_test/sign_mnist_test.csv", "rb"), delimiter=",", skiprows=1)
Tlabels = Tmatrix[:,0]
Tmatrix = Tmatrix[:,1:]

matrix = np.loadtxt(open("sign_mnist_train/sign_mnist_train.csv", "rb"), delimiter=",", skiprows=1)
labels = matrix[:,0]
matrix = matrix[:,1:]

X = pd.DataFrame(matrix)
y = pd.Series(labels)
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0,
#                                                     random_state=1234123)

print(matrix.shape)
print(labels.shape)
print(pd.Series(y).value_counts())

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X, y)
Ty_pred = svm.predict(Tmatrix)

# calculate accuracy
accuracy = accuracy_score(Tlabels, Ty_pred)
print('Model accuracy is: ', accuracy)
