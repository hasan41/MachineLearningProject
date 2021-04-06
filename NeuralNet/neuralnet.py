import os

import matplotlib as mpl
import matplotlib.pyplot as plt
# from IPython.display import display

import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split


train = pd.read_csv('/Users/hasannaseer/Downloads/MINST/sign_mnist_train.csv')
test = pd.read_csv('/Users/hasannaseer/Downloads/MINST/sign_mnist_test.csv')

#storing labels
labels = train['label'].values
images = train.drop('label', axis=1)
images = images.values
images = np.array([np.reshape(x, (28, 28)) for x in images])
images = np.array([x.flatten() for x in images])

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)

print(x_train.shape)
print(x_test.shape)


x_train = x_train / 255
x_test = x_test / 255