#import data sets for sign language
import string
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#accessing data sets

df = pd.read_csv('/content/sign_mnist_train.csv')
df.head()
def load_data(path):
	df = pd.read_csv(path)
	y = np.array([label if label < 9
				else label-1 for label in df['label']])
	df = df.drop('label', axis=1)
	x = np.array([df.iloc[i].to_numpy().reshape((28, 28))
				for i in range(len(df))]).astype(float)
	x = np.expand_dims(x, axis=3)
	y = pd.get_dummies(y).values

	return x, y

X_train, Y_train = load_data('/content/sign_mnist_train.csv')
X_test, Y_test = load_data('/content/sign_mnist_test.csv')

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
