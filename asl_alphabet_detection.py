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
