import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from traitement import g,traitement_data,u,v,nettoyage_y

density_file='density_table.csv'
df0=traitement_data(density_file)
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
data_std=(data - data.mean()) / data.std(ddof = 0)

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


from sklearn.model_selection import KFold
kf=KFold(n_splits=2,shuffle=True)

def create_evaluate_model(index_fold,x_train,x_test,y_train,y_test):
    regression_alg=LinearRegression()
    regression_alg.fit(x_train,y_train)
    test_predictions=regression_alg.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,test_predictions))
    r2=r2_score(y_test,test_predictions)
    print(f"Run {index_fold}:RMSE={round(rmse,2)}-R2_score={round(r2,2)}")
    return (rmse,r2)

print(kf)