import csv
import pandas as pd
import numpy as np
import io
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

df_ACC_TRA = pd.read_csv('Data/Accidentes_de_transito_en_carreteras-2020-2021-Sutran.csv', encoding='utf-8-sig', delimiter=';')

columnaCodigoVia = []

# Configuración de pandas para mostrar todas las columnas y ajustar el ancho
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Mostrar las primeras 5 filas del DataFrame
print("Primera vista del DataFrame original:")
print(df_ACC_TRA.head(100).to_string(index=False))

# Configuración de pandas para mostrar todas las columnas y ajustar el ancho
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Definir las columnas a eliminar basándonos en los nombres exactos impresos
DROP_COLUMNS = ['FECHA_CORTE', 'FECHA']

# Eliminar las columnas especificadas
df_ACC_TRA.drop(columns=DROP_COLUMNS, inplace=True)

# Listado donde están almacenados los campos relacionados al dataset de accidentes de tránsito
columnas_ACC_TRA = list(df_ACC_TRA.select_dtypes(include=['object']).columns)

def convertir_horas_a_minutos(tiempo):
    if tiempo == 'N.I.':
        return -1
    try:
        horas, minutos = map(int, tiempo.split(':'))
        total_minutos = horas * 60 + minutos
        return total_minutos
    except ValueError:
        return -1

def procesar_datos():
    global df_ACC_TRA, columnaCodigoVia
    

    # Crear nueva columna de hora en minutos 
    df_ACC_TRA["HORA_MINUTOS"] = df_ACC_TRA["HORA"].apply(convertir_horas_a_minutos)

    # Realizar One-Hot encoding para la hora en minutos, con esto tendremos separados la hora en diferentes categorias
    df_ACC_TRA["HORA_N.I."] = pd.cut(x = df_ACC_TRA["HORA_MINUTOS"],
                                                    bins = [-2, 0, 360, 720, 1140, 1440],
                                                    labels = [1, 0, 0, 0, 0],ordered=False)
    df_ACC_TRA["HORA_TEMPRANO"] = pd.cut(x = df_ACC_TRA["HORA_MINUTOS"],
                                                    bins = [-2, 0, 360, 720, 1140, 1440],
                                                    labels = [0, 0, 1, 0, 0],ordered=False)
    df_ACC_TRA["HORA_TARDE"] = pd.cut(x = df_ACC_TRA["HORA_MINUTOS"],
                                                    bins = [-2, 0, 360, 720, 1140, 1440],
                                                    labels = [0, 0, 0, 1, 0],ordered=False)
    df_ACC_TRA["HORA_NOCHE"] = pd.cut(x = df_ACC_TRA["HORA_MINUTOS"],
                                                    bins = [-2, 0, 360, 720, 1140, 1440],
                                                    labels = [0, 0, 0, 0, 1], ordered=False)
    df_ACC_TRA["HORA_MADRUGADA"] = pd.cut(x = df_ACC_TRA["HORA_MINUTOS"],
                                                    bins = [-2, 0, 360, 720, 1140, 1440],
                                                    labels = [0, 1, 0, 0, 0], ordered=False)
    # Eliminamos los campos innecesarios como Hora y hora minutos
    df_ACC_TRA.drop(columns = ['HORA','HORA_MINUTOS'], inplace=True)

    # Almacenar en una lista los registros del código de vía sin repetir los datos
    columnaCodigoVia = list(df_ACC_TRA['CODIGO_VIA'].value_counts().index)

    # Eliminar registros que sean duplicados
    df_ACC_TRA = df_ACC_TRA.drop_duplicates() if df_ACC_TRA.duplicated().any() else df_ACC_TRA

    # Almacenar en un diccionario los códigos de vía, en el cual serán enumerados del 1 en adelante
    diccionario_codigo_via = {element: index + 1 for index, element in enumerate(columnaCodigoVia)}

    # Convertir la columna de CODIGO_VÍA que está en cadena en un label encoded data
    df_ACC_TRA["CODIGO_VIA"] = df_ACC_TRA["CODIGO_VIA"].map(diccionario_codigo_via)

    # Existencias de departamentos en minúsculas, por lo que forzamos las mayúsculas
    df_ACC_TRA['DEPARTAMENTO'] = df_ACC_TRA['DEPARTAMENTO'].str.upper()
    
    # Aplicar Label Encoding a la columna 'DEPARTAMENTO'
    label_encoder = LabelEncoder()
    df_ACC_TRA['DEPARTAMENTO'] = label_encoder.fit_transform(df_ACC_TRA['DEPARTAMENTO'])

    # Aplicar One-Hot Encoding a los campos 'MODALIDAD'
    df_ACC_TRA = pd.get_dummies(df_ACC_TRA, columns=['MODALIDAD'], prefix=['TIPO'])

    # Convertir solo las columnas de One-Hot Encoding a valores enteros (0 y 1)
    for column in df_ACC_TRA.columns:
        if 'TIPO_' in column:
            df_ACC_TRA[column] = df_ACC_TRA[column].astype(int)

procesar_datos()

# Mostrar las primeras 100 filas para verificar el resultado final
print("\nVista del DataFrame después de todas las transformaciones:")
print(df_ACC_TRA.head(100).to_string(index=False))
