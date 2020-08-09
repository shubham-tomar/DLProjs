import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('Churn_Modelling.csv')
print(df.head)     

X = df.iloc[:,3:13]
y = df.iloc[:,13]
geography = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)
X = pd.concat([X,geography,gender],axis=1)
X = X.drop(["Geography","Gender"],axis=1)
# print(X.isnull().sum())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

