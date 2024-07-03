import csv
import pandas as pd
import numpy as np
import io
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import mode

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

# Contar las etiquetas después de limpiar los NaN
num_filas = df_ACC_TRA.shape[0]
# Mostrar el conteo de filas
print("Conteo de filas antes de eliminar los N.I.:")
print(num_filas)

# Eliminar filas donde alguna de estas columnas contiene "N.I."
columns_to_check = ['HORA', 'DEPARTAMENTO', 'CODIGO_VIA', 'KILOMETRO', 'MODALIDAD', 'FALLECIDOS', 'HERIDOS']
df_ACC_TRA = df_ACC_TRA[~df_ACC_TRA[columns_to_check].isin(['N.I.']).any(axis=1)]

# Contar las etiquetas después de limpiar los NaN
num_filas = df_ACC_TRA.shape[0]
# Mostrar el conteo de filas
print("Conteo de filas después de eliminar los N.I.:")
print(num_filas)

def convertir_horas_a_minutos(tiempo):
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

# Normalización de los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_ACC_TRA.select_dtypes(include=[np.number]))

# Determinar el número óptimo de clusters utilizando el método del codo
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Aquí se agrega el valor explícito de n_init
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('SSE (Inercia)')
plt.title('Método del codo')
plt.savefig('metodo_del_codo.png')
plt.show()

# Aplicacion de K-Means con el número de clusters seleccionado
clusters = 5
kmeans = KMeans(n_clusters = clusters, n_init = 10)
kmeans.fit(df_scaled)

# Añadir los clusters al DataFrame original
df_ACC_TRA['Cluster'] = kmeans.labels_

print(df_ACC_TRA.head(100).to_string(index=False))

# Asegurar que las columnas fallecidos y heridos sean numéricas
df_ACC_TRA['FALLECIDOS'] = pd.to_numeric(df_ACC_TRA['FALLECIDOS'])
df_ACC_TRA['HERIDOS'] = pd.to_numeric(df_ACC_TRA['HERIDOS'])


# Calcular estadísticas descriptivas por cluster
cluster_count = df_ACC_TRA.groupby('Cluster').size()
cluster_analysis = df_ACC_TRA.groupby('Cluster').agg({
    'DEPARTAMENTO': lambda x: x.mode().iloc[0],
    'KILOMETRO': lambda x: x.mode().iloc[0],
    'CODIGO_VIA': lambda x: x.mode().iloc[0],
    'FALLECIDOS': 'mean',
    'HERIDOS': 'mean',
    'HORA_TEMPRANO': lambda x: x.mode().iloc[0],
    'HORA_TARDE': lambda x: x.mode().iloc[0],
    'HORA_NOCHE': lambda x: x.mode().iloc[0],
    'HORA_MADRUGADA': lambda x: x.mode().iloc[0],
    'TIPO_ATROPELLO': lambda x: x.mode().iloc[0],
    'TIPO_CHOQUE': lambda x: x.mode().iloc[0],
    'TIPO_DESPISTE': lambda x: x.mode().iloc[0],
    'TIPO_ESPECIAL': lambda x: x.mode().iloc[0],
    'TIPO_VOLCADURA': lambda x: x.mode().iloc[0]
})

print("\nAnálisis de los clusters:")
print(cluster_count)
print(cluster_analysis)

# Realizar el groupby y calcular la media por cluster
cluster_analysis = df_ACC_TRA.groupby('Cluster')[['FALLECIDOS', 'HERIDOS']].mean()

# Visualización y análisis de resultados
plt.figure(figsize=(10, 6))
cluster_analysis.plot(kind='bar', stacked=True, colormap='viridis')
plt.xlabel('Cluster')
plt.ylabel('Promedio')
plt.title('Comparación de Fallecidos y Heridos por Cluster')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Mostrar las características comunes de los clusters con mayor número de heridos y fallecidos
cluster_max_heridos = cluster_analysis.nlargest(1, 'HERIDOS')
cluster_max_fallecidos = cluster_analysis.nlargest(1, 'FALLECIDOS')

print("\nCluster con Mayor Número de Heridos:")
print(cluster_max_heridos)
print("\nCluster con Mayor Número de Fallecidos:")
print(cluster_max_fallecidos)
