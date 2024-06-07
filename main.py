import csv
import pandas as pd
import numpy as np
import io
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df_ACC_TRA = pd.read_csv('Data\Accidentes_de_transito_en_carreteras-2020-2021-Sutran.csv', encoding='utf8', delimiter=';')

df_ACC_TRA.head(100)

print(df_ACC_TRA.head(100))