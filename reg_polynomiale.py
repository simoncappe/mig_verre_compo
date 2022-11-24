import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



density_file='density_table.csv'

#oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
#extract_table(density_file,oxides,'Density')
#select_data('density_100_tableonly',oxides,'Density')
#cleanup_data('Density_100_tableonly_data',0.1,100)
df0=pd.read_csv('fracture_toughness_100.csv')
grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]

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
polynomial_features=PolynomialFeatures(degree=2)
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

def create_evaluate_model_polynomiale(index_fold,x_train,x_test,y_train,y_test,deg):
    polynomial_features=PolynomialFeatures(degree=deg)
    poly_regression_alg=LinearRegression()

    model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)])

    regression_alg=LinearRegression()
    model.fit(x_train,y_train)
    test_predictions=model.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,test_predictions))
    r2=r2_score(y_test,test_predictions)
    print(f"Run {index_fold}:RMSE={round(rmse,2)}-R2_score={round(r2,2)}")
    return (rmse,r2)

nb_model=5
kf=KFold(n_splits=nb_model, shuffle=False)

index_fold=0
average_rmse=0
average_r2=0
deg=2
for train_index, test_index in kf.split(data):
    x_train, x_test=data.iloc[train_index], data.iloc[test_index]
    y_train, y_test=y.iloc[train_index], y.iloc[test_index]
    
    current_rmse,current_r2=create_evaluate_model_polynomiale(index_fold,x_train,x_test,y_train,y_test,deg)

    average_rmse=average_rmse+current_rmse
    average_r2=average_r2+current_r2

    index_fold= index_fold +1
average_rmse=average_rmse/nb_model
average_r2=average_r2/nb_model
print(f"Moyenne : RMSE={round(average_rmse,2)}-R2-score={round(average_r2,2)}")
