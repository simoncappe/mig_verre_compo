import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from traitement_backup import traitement_data,u,v,nettoyage_y,traitement_data_improved

liquid0='liquidus_table_100.csv'
liquid9='liquidus_table_95.csv'
density_file='density_table.csv'
frac='fracture_toughness_100.csv'
visc='viscosity_table.csv'

