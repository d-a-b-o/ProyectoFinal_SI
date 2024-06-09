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

# Configuración de pandas para mostrar todas las columnas y ajustar el ancho
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Mostrar las primeras 5 filas del DataFrame
print("Primera vista del DataFrame original:")
print(df_ACC_TRA.head(100).to_string(index=False))

# Definir las columnas a eliminar basándonos en los nombres exactos impresos
DROP_COLUMNS = ['FECHA_CORTE', 'FECHA', 'KILOMETRO', 'FALLECIDOS', 'HERIDOS']

# Eliminar las columnas especificadas
df_ACC_TRA.drop(columns=DROP_COLUMNS, inplace=True)

# Mostrar las primeras 5 filas del DataFrame después de eliminar las columnas
print("\nVista del DataFrame después de eliminar columnas:")
print(df_ACC_TRA.head(100).to_string(index=False))

# Aplicar One-Hot Encoding al campo 'MODALIDAD'
df_one_hot_modalidad = pd.get_dummies(df_ACC_TRA, columns=['MODALIDAD'])

# Convertir solo las columnas de One-Hot Encoding a valores enteros (0 y 1)
for column in df_one_hot_modalidad.columns:
    if 'MODALIDAD_' in column:
        df_one_hot_modalidad[column] = df_one_hot_modalidad[column].astype(int)

# Mostrar las primeras 5 filas para verificar el resultado
print("\nVista del DataFrame después de One-Hot Encoding:")
print(df_one_hot_modalidad.head(100).to_string(index=False))
