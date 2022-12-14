import pandas as pd
import numpy as np
#traitement liquidus
df=pd.read_csv('liquidus_table_95.csv',usecols=[
       ' SiO2', ' Al2O3', ' MgO', ' CaO', ' Na2O',
       ' Liquidus Temperature ( C )'])
v=df[' Al2O3'][3]
df.replace(v,0.,inplace=True)
df=df.astype(np.float64)
df['sum']=np.sum(df.iloc[:,0:5],axis=1)
df=df[abs(df['sum']-100)<10**(-3)]
#traitement toughness
f=pd.read_csv('fracture_toughness_100.csv',
usecols=[' SiO2', ' Al2O3', ' MgO', ' CaO', ' Na2O',
       ' Fracture Toughness ( MPa.m1/2 )'])
u=f[' Al2O3'][0]
print(f[' Al2O3'][0])
f.replace(u,0.,inplace=True)
f=f.astype(np.float64)
f['sum']=np.sum(f.iloc[:,0:5],axis=1)
f=f[abs(f['sum']-100)<10**(-5)]
#traitement density
g=pd.read_csv('density_table.csv',usecols=[' SiO2', ' Al2O3', ' MgO', ' CaO',
' Na2O', ' Density ( g/cm3 )'])
g.replace([u,v],[float(0.),float(0.)],inplace=True)
g=g.astype(np.float64,copy=True)

g['sum']=np.sum(g.iloc[:,0:5],axis=1)
g=g[abs(g['sum']-100)<10**(-5)]


def traitement_data(file:str):
    df_bis= pd.read_csv(file)
    l=list(df_bis.columns)
    r=pd.read_csv(file,usecols=l[5:11])
    p=r.columns
    r.replace([u,v],[0.,0.],inplace=True)
    df0=r.astype(float,copy=True)
    df0['sum']=np.sum(df0.iloc[:,0:5],axis=1)
    df0=df0[abs(df0['sum']-100)<10**(-3)]
    df0=df0[p]
    return df0
def traitement_data_improved(file:str):
    df_bis= pd.read_csv(file)
    l=list(df_bis.columns)
    r=pd.read_csv(file,usecols=l[5:12])
    p=r.columns
    r.replace([u,v],[0.,0.],inplace=True)
    df0=r.astype(float,copy=True)
    df0['sum']=np.sum(df0.iloc[:,0:5],axis=1)
    df0=df0[abs(df0['sum']-100)<10**(-3)]
    df0=df0[p]
    return df0

def nettoyage_y(data):
       y=data.columns[-1]
       m=data[y].mean()
       e=np.sqrt(data[y].var())
       data=data[abs(data[y]-m)<=e]
       return data

h=pd.read_csv('viscosity_table.csv',usecols=[' SiO2', ' Al2O3', 
' MgO', ' CaO', ' Na2O', ' SO3',
' Viscocity at 1000C ( dPa.s )'])
l=h.columns
w=h[' MgO'][794]
h.replace([u,v,w],[0.,0.,0.],inplace=True)
h=h.astype(np.float64)
h['sum']=np.sum(h.iloc[:,0:6],axis=1)
h=h[abs(h['sum']-100)<10**(-5)]
h=h[l]
