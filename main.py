import csv
import pandas as pd
import numpy as np
import io
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

df_ACC_TRA = pd.read_csv('Data/Accidentes_de_transito_en_carreteras-2020-2021-Sutran.csv', encoding='utf8', delimiter=';')

columnaCodigoVia = []

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

# Listado donde estan almacenados los campos relacionado al dataset de accidentes de transito
columnas_ACC_TRA = list(df_ACC_TRA.select_dtypes(include=['object']).columns)

def procesar_datos():
    global df_ACC_TRA, columnaCodigoVia

    # Almacenar en una lista los registros del codigo de via sin repetir los datos
    columnaCodigoVia = list(df_ACC_TRA['CODIGO_VIA'].value_counts().index)

    # Eliminar registros que sean duplicados
    df_ACC_TRA = df_ACC_TRA.drop_duplicates() if df_ACC_TRA.duplicated().any() else df_ACC_TRA

    # Almacenar en un diccionario los codigos via, en el cual serán enumerados del 1 al 175
    diccionario_codigo_via = {element: index+1 for index, element in enumerate(columnaCodigoVia)}
    # Convertir la columna de codigo_via que estan en cadena en un label encoded data
    df_ACC_TRA["CODIGO_VIA"] = df_ACC_TRA["CODIGO_VIA"].map(diccionario_codigo_via)

    # Existencias de departamentos en minusculas, por lo que forzamos las mayusculas
    df_ACC_TRA['DEPARTAMENTO'] = df_ACC_TRA['DEPARTAMENTO'].str.upper()
    
    # Aplicar One-Hot Encoding a los campos 'MODALIDAD' y 'DEPARTAMENTO'
    df_ACC_TRA = pd.get_dummies(df_ACC_TRA, columns=['MODALIDAD', 'DEPARTAMENTO'])

    # Convertir solo las columnas de One-Hot Encoding a valores enteros (0 y 1)
    for column in df_ACC_TRA.columns:
        if 'MODALIDAD_' in column or 'DEPARTAMENTO_' in column:
            df_ACC_TRA[column] = df_ACC_TRA[column].astype(int)

procesar_datos()
# Mostrar las primeras 100 filas para verificar el resultado
print("\nVista del DataFrame después de One-Hot Encoding:")
print(df_ACC_TRA.head(100).to_string(index=False))
