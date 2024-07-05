from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Definición de constantes
CSV_FILE_PATH = "Data/Accidentes_de_transito_en_carreteras-2020-2021-Sutran.csv"
DROP_COLUMNS = ["FECHA_CORTE", "FECHA"]
COLUMNS_TO_CHECK = [
    "HORA",
    "DEPARTAMENTO",
    "CODIGO_VIA",
    "KILOMETRO",
    "MODALIDAD",
    "FALLECIDOS",
    "HERIDOS",
]
HORA_BINS = [-2, 0, 360, 720, 1140, 1440]
HORA_LABELS = {
    "HORA_TEMPRANO": [0, 0, 1, 0, 0],
    "HORA_TARDE": [0, 0, 0, 1, 0],
    "HORA_NOCHE": [0, 0, 0, 0, 1],
    "HORA_MADRUGADA": [0, 1, 0, 0, 0],
}
CLUSTERS = 5

"""Carga los datos desde un archivo CSV."""
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path, encoding="utf-8-sig", delimiter=";")
    print("Datos cargados exitosamente")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("Primera vista del DataFrame original:")
    print(df.head(100).to_string(index=False))
    return df

"""Limpia los datos eliminando columnas innecesarias y filas con valores 'N.I.'."""
def clean_data(df):
    df.drop(columns=DROP_COLUMNS, inplace=True)
    print("Conteo de filas antes de eliminar los N.I.:")
    print(df.shape[0])
    df = df[~df[COLUMNS_TO_CHECK].isin(["N.I."]).any(axis=1)]
    print("Conteo de filas después de eliminar los N.I.:")
    print(df.shape[0])
    return df

"""Convierte una hora en formato 'HH:MM' a minutos desde la medianoche."""
def convertir_horas_a_minutos(tiempo):
    try:
        horas, minutos = map(int, tiempo.split(":"))
        return horas * 60 + minutos
    except ValueError:
        return -1

"""Crea características de tiempo basadas en la columna 'HORA'."""
def create_time_features(df):
    df["HORA_MINUTOS"] = df["HORA"].apply(convertir_horas_a_minutos)
    for feature, labels in HORA_LABELS.items():
        df[feature] = pd.cut(
            df["HORA_MINUTOS"], bins=HORA_BINS, labels=labels, ordered=False
        )
    df.drop(columns=["HORA", "HORA_MINUTOS"], inplace=True)
    return df

"""Codifica características categóricas en el DataFrame."""
def encode_features(df):
    # Mapeo de códigos de vías
    columnaCodigoVía = list(df["CODIGO_VIA"].value_counts().index)
    diccionario_codigo_via = {
        element: index + 1 for index, element in enumerate(columnaCodigoVía)
    }
    df["CODIGO_VIA"] = df["CODIGO_VIA"].map(diccionario_codigo_via)

    # Codificación de departamentos
    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].str.upper()
    label_encoder = LabelEncoder()
    df["DEPARTAMENTO"] = label_encoder.fit_transform(df["DEPARTAMENTO"])

    # One-hot encoding para la modalidad
    df = pd.get_dummies(df, columns=["MODALIDAD"], prefix=["TIPO"])

    # Convertir columnas one-hot a enteros
    for column in df.columns:
        if "TIPO_" in column:
            df[column] = df[column].astype(int)
    return df

"""Aplica todas las funciones de limpieza y codificación al DataFrame."""
def preprocess_data(df):
    df = clean_data(df)
    df = create_time_features(df)
    df = encode_features(df)
    return df

"""Genera un gráfico del método del codo para determinar el número óptimo de clusters."""
def plot_elbow_method(df_scaled):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), sse, marker="o")
    plt.xlabel("Número de clusters")
    plt.ylabel("SSE (Inercia)")
    plt.title("Método del codo")
    plt.savefig("metodo_del_codo.png")
    plt.show()

"""Aplica el algoritmo K-means al DataFrame escalado."""
def apply_kmeans(df, df_scaled, clusters):
    kmeans = KMeans(n_clusters=clusters, n_init=10)
    kmeans.fit(df_scaled)
    df["Cluster"] = kmeans.labels_
    return df, kmeans

"""Genera un gráfico 3D de los clusters."""
def plot_cluster_3d(df, kmeans):
    colormap = np.array(["red", "green", "blue", "yellow", "black", "pink"])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df["DEPARTAMENTO"], df["FALLECIDOS"], df["HERIDOS"], c=colormap[kmeans.labels_]
    )
    ax.set_xlabel("Departamento")
    ax.set_ylabel("Fallecidos")
    ax.set_zlabel("Heridos")
    ax.set_title("Agrupación K-means en 3D")
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    plt.show()

"""Genera un gráfico de barras para comparar fallecidos y heridos por cluster."""
def plot_cluster_analysis(df, cluster_analysis):
    cluster_analysis = df.groupby("Cluster")[["FALLECIDOS", "HERIDOS"]].mean()
    plt.figure(figsize=(10, 6))
    cluster_analysis.plot(kind="bar", stacked=True, colormap="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("Promedio")
    plt.title("Comparación de Fallecidos y Heridos por Cluster")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

"""Realiza un análisis de la silueta para diferentes números de clusters."""
def silhouette_analysis_nclusters(df_scaled):
    silhouette_scores = []
    range_n_clusters = range(2, 11)
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        silhouette_avg = silhouette_score(df_scaled, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Para n_clusters = {n_clusters}, el índice de Silhouette promedio es: {silhouette_avg}")
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, silhouette_scores, marker="o")
    plt.title("Índice de Silhouette para diferentes valores de n_clusters")
    plt.xlabel("Número de clusters (n_clusters)")
    plt.ylabel("Índice de Silhouette promedio")
    plt.grid(True)
    plt.show()

"""Genera un gráfico de análisis de silueta para el número de clusters especificado."""
def plot_silhouette_analysis(kmeans, df_scaled, clusters):
    labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, labels)
    print(f"El índice de Silhouette promedio para n_clusters={clusters} es: {silhouette_avg}")
    sample_silhouette_vals = silhouette_samples(df_scaled, labels)
    plt.figure()
    y_lower, y_upper = 0, 0
    for i in range(clusters):
        cluster_silhouette_vals = sample_silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor="none", height=1)
        y_lower += len(cluster_silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--")
    plt.xlabel("Coeficiente de Silueta")
    plt.ylabel("Clúster")
    plt.title("Análisis de Silueta")
    plt.savefig("analisis_de_silueta_final.png")
    plt.show()

"""Analiza los clusters generados para encontrar características comunes."""
def analyze_clusters(df):
    cluster_analysis = df.groupby("Cluster").agg({
        "DEPARTAMENTO": lambda x: x.mode().iloc[0],
        "KILOMETRO": lambda x: x.mode().iloc[0],
        "CODIGO_VIA": lambda x: x.mode().iloc[0],
        "FALLECIDOS": "mean",
        "HERIDOS": "mean",
        "HORA_TEMPRANO": lambda x: x.mode().iloc[0],
        "HORA_TARDE": lambda x: x.mode().iloc[0],
        "HORA_NOCHE": lambda x: x.mode().iloc[0],
        "HORA_MADRUGADA": lambda x: x.mode().iloc[0],
        "TIPO_ATROPELLO": lambda x: x.mode().iloc[0],
        "TIPO_CHOQUE": lambda x: x.mode().iloc[0],
        "TIPO_DESPISTE": lambda x: x.mode().iloc[0],
        "TIPO_ESPECIAL": lambda x: x.mode().iloc[0],
        "TIPO_VOLCADURA": lambda x: x.mode().iloc[0],
    })
    return cluster_analysis

"""Imprime las características comunes de los clusters con mayor número de heridos y fallecidos."""
def max_heridos_fallecidos(df, cluster_analysis):
    cluster_max_heridos = df.loc[df['Cluster'] == cluster_analysis['HERIDOS'].idxmax()]
    cluster_max_fallecidos = df.loc[df['Cluster'] == cluster_analysis['FALLECIDOS'].idxmax()]
    print("\nCaracterísticas comunes del cluster con mayor número de heridos:")
    print(cluster_max_heridos.describe(include='all'))
    print("\nCaracterísticas comunes del cluster con mayor número de fallecidos:")
    print(cluster_max_fallecidos.describe(include='all'))

"""Genera un gráfico de barras para la frecuencia de accidentes por departamento."""
def frecuencia_accidentes_departamento(df):
    frecuencia_departamento = df['DEPARTAMENTO'].value_counts()
    print("\nFrecuencia de accidentes por departamento:")
    print(frecuencia_departamento)
    plt.figure(figsize=(12, 8))
    frecuencia_departamento.plot(kind='bar', color='skyblue')
    plt.title('Frecuencia de Accidentes por Departamento')
    plt.xlabel('Departamento')
    plt.ylabel('Número de Accidentes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
"""Genera un gráfico de barras para la influencia de la hora del accidente."""
def influencia_hora_accidente(df):
    df['HORA_TEMPRANO'] = df['HORA_TEMPRANO'].astype(int)
    df['HORA_TARDE'] = df['HORA_TARDE'].astype(int)
    df['HORA_NOCHE'] = df['HORA_NOCHE'].astype(int)
    df['HORA_MADRUGADA'] = df['HORA_MADRUGADA'].astype(int)
    hora_accidente = df[['HORA_TEMPRANO', 'HORA_TARDE', 'HORA_NOCHE', 'HORA_MADRUGADA']].sum()
    print("\nInfluencia de la hora del accidente:")
    print(hora_accidente)
    plt.figure(figsize=(8, 6))
    hora_accidente.plot(kind='bar', color='lightgreen')
    plt.title('Influencia de la Hora del Accidente')
    plt.xlabel('Hora del Día')
    plt.ylabel('Número de Accidentes')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
"""Genera un gráfico de barras para la influencia de la modalidad del accidente."""
def influencia_modalidad_accidente(df):
    modalidad_accidente = df.filter(like='TIPO_').sum()
    print("\nInfluencia de la modalidad del accidente:")
    print(modalidad_accidente)
    plt.figure(figsize=(10, 6))
    modalidad_accidente.plot(kind='bar', color='salmon')
    plt.title('Influencia de la Modalidad del Accidente')
    plt.xlabel('Modalidad')
    plt.ylabel('Número de Accidentes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

"""Función principal que ejecuta todo el flujo de trabajo."""
def main():
    # Cargar y preprocesar datos
    df = load_data(CSV_FILE_PATH)
    df = preprocess_data(df)
    print("\nVista del DataFrame después de todas las transformaciones:")
    print(df.head(100).to_string(index=False))

    # Escalar datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))

    # Método del codo para determinar el número óptimo de clusters
    plot_elbow_method(df_scaled)

    # Aplicar K-means
    df, kmeans = apply_kmeans(df, df_scaled, CLUSTERS)
    print(df.head(100).to_string(index=False))

    # Convertir columnas de fallecidos y heridos a numéricas
    df["FALLECIDOS"] = pd.to_numeric(df["FALLECIDOS"])
    df["HERIDOS"] = pd.to_numeric(df["HERIDOS"])

    # Analizar clusters
    cluster_analysis = analyze_clusters(df)
    print("\nAnálisis de los clusters:")
    print(cluster_analysis)

    # Generar gráficos y análisis adicionales
    plot_cluster_analysis(df, cluster_analysis)
    max_heridos_fallecidos(df, cluster_analysis)
    plot_cluster_3d(df, kmeans)
    silhouette_analysis_nclusters(df_scaled)
    plot_silhouette_analysis(kmeans, df_scaled, CLUSTERS)

    # Análisis de frecuencia y modalidades de accidentes
    frecuencia_accidentes_departamento(df)
    influencia_hora_accidente(df)
    influencia_modalidad_accidente(df)

if __name__ == "__main__":
    main()
