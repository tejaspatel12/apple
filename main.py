#Import Libraries
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from matplotlib import animation, colors
import pandas as pd

# value1 = input("Enter your value: ")
# print("Your value is :"+ value1)

filename = 'data.txt'

# banknote authentication Data Set
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
# Irvine, CA: University of California, School of Information and Computer Science.

data_file = filename
data_x = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=range(0,2) ,dtype=np.float64)
data_y = np.loadtxt(data_file, delimiter=",", skiprows=0, usecols=(2,),dtype=np.int64)

df = pd.read_csv(filename, sep =",")
df.head(20)

# train and test split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes

# print("Hellow")
