import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from traitement import traitement_data,u,v,nettoyage_y

liquid0='liquidus_table_100.csv'
liquid9='liquidus_table_95.csv'
density_file='density_table.csv'
frac='fracture_toughness_100.csv'

df0=traitement_data(frac)
df0=nettoyage_y(df0)
grandeur_mesuree=df0.columns[-1]
data=df0[[' SiO2', ' Al2O3', ' MgO', ' CaO',
' Na2O']]

y=df0[[grandeur_mesuree]]
#
def normalization(data):
    cols=data.columns
    print(cols)
    for col in data:
        x=data.mean[[col]].values.astype(float)
        std_n=preprocessing.StandardScaler()
        res=std_n.fit_transform(x)
        data[col]=res
#
data=(data - data.mean()) / data.std(ddof = 0)

x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
polynomial_features=PolynomialFeatures(degree=5)
poly_regression_alg=LinearRegression()

model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
model.fit(x_train,y_train)

train_predictions=model.predict(x_train)

print(f"RMSE= {round(np.sqrt(mean_squared_error(y_train,train_predictions)),2)}")
print(f"R2_score={round(r2_score(y_train,train_predictions),2)}")


