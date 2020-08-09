import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('Churn_Modelling.csv')
print(df.head)     

X = df.iloc[:,3:13]
y = df.iloc[:,13]

