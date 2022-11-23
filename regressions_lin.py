import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from traitement import traitement_data,u,v


density_file='density_table.csv'
df0=traitement_data(density_file)
df=df0[[' SiO2', ' Al2O3', ' MgO', ' CaO',
' Na2O', ' Density ( g/cm3 )']]
data=df0[[' SiO2', ' Al2O3', ' MgO', ' CaO',
' Na2O']]
density=df0[[' Density ( g/cm3 )']]
def normalization(data):
    cols=data.columns
    print(cols)
    for col in data:
        x=data.mean[[col]].values.astype(float)
        std_n=preprocessing.StandardScaler()
        res=std_n.fit_transform(x)
        data[col]=res

data_std=(data - data.mean()) / data.std(ddof = 0)

x_train,x_test,y_train,y_test=train_test_split(data,density,test_size=0.2)

regression_alg=LinearRegression()
regression_alg.fit(x_train,y_train)

train_predictions=regression_alg.predict(x_train)

print(f"RMSE= {round(np.sqrt(mean_squared_error(y_train,train_predictions)),2)}")
print(f"R2_score={round(r2_score(y_train,train_predictions),2)}")
