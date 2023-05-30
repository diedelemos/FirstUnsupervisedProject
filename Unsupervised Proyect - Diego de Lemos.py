# Diego Rafael de Lemos Burgaña

# Unsupervised Project - Machine Learning

# Primero cargamos las librerias necesarias para hacer el trabajo

import pandas as pd
import numpy as np
import io
import requests
import csv
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Importamos los datos descargados de internet. 

# En este trabajo de Machine Learning Unsupervised vamos a utilizar los datos de la temporada de baloncesto del 2013 de la NBA en donde se encuentran distintos analisis sobre los jugadores y su impacto al jugar.

nba = pd.read_csv(r"C:\Users\diede\OneDrive\Desktop\Trabajos\Machine Learning\Machine Learning Project\Unsupervised\nba_2013.csv", encoding='cp1252', index_col=None)

# Vemos los datos
print(nba)
nba.head

# Limpiamos y preparamos los datos
# Eliminamos las variables que no aportan nada a la investigacion
nba.drop(nba.columns[[29,30]], axis=1, inplace=True)
print(nba)

# Vemos los tipos de datos por variable
nba.dtypes

# Vemos la cantidad de NA que hay en cada columna del dataframe
nba.isna().sum()
nba = nba.dropna(axis=0, how='any')

# Seleccionamos solo las variables numéricas
nba = nba.select_dtypes(include=np.number)
nba.head

# Creamos 2 nuevas variables, ppg que mide los puntos por partido para cada jugador y atr que mide la proporcion de asistencias por cada perdida de balon de jugador.
nba["ppg"] = nba["pts"]/nba["g"]
nba = nba[nba["tov"] != 0].copy()
nba["atr"] = nba["ast"]/nba["tov"]

nba[["ppg","atr"]].head()

nba.index
nba.shape[0]

# Utilizo el elbow method para encontrar la cantidad de cluster mas optima
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(nba)
    distortions.append(kmeanModel.inertia_)

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

# Definimos los cluster, he decidido utilizar 4
cluster = 4
cluster1 = 2
np.random.seed(1)
random_initial_value = np.random.choice(nba.index, size = cluster)
random_initial_value

nbacluster = nba.loc[random_initial_value]
nbacluster


def centroids_to_dic(centroids):
    dictionary = {}
    counter = 0
    for index, row in nbacluster.iterrows():
        coordinates = [row["ppg"], row["atr"]]
        dictionary[counter] = coordinates
        counter += 1
    return dictionary

centroids_dic = centroids_to_dic(nbacluster)
centroids_dic

nba.iloc[0][["ppg", "atr"]]

# Calulamos las distancias
def calculate_distance(q,p):
    distance = 0
    for i in range(len(q)):
        distance += (q[i] - p[i])**2
    return np.sqrt(distance)

q = [6.87, 3.01]
p = [20.70,0.57]
print(calculate_distance(q,p))

row1_distances = []
for q1 in centroids_dic.values():
    distance = calculate_distance(q1,p)
    row1_distances.append(distance)

row1_distances

minimum = min(row1_distances)
row1_distances.index(minimum)

# Definimos los clusters por jugador
def assign_cluster(row):
    row_distances = []
    p = [row["ppg"], row["atr"]]
    for q in centroids_dic.values():
        distance = calculate_distance(q,p)
        row_distances.append(distance)
    minimum = min(row_distances)
    cluster = row_distances.index(minimum)
    return cluster

assign_cluster(nba.iloc[0])

# Añadimos la variable cluster a el dataframe
nba["cluster"] = nba.apply(lambda row :assign_cluster(row),axis = 1)
nba["cluster"].value_counts()

# Visualizamos la distribucion de los distintos clusters por jugador
def visualize_cluster(df, num_cluster):
    colors = ["black","green", "blue", "orange", "purple"]
    for i in range(num_cluster):
        cluster = df[df["cluster"] == i]
        plt.scatter(cluster["ppg"], cluster["atr"], c = colors[i])
        plt.xlabel("Points Per Game")
        plt.ylabel("Assits Turnover Ratio")
    plt.show()

visualize_cluster(nba, cluster)


# Otro metodo es utilizando el algoritmo KMeans como vimos en clase, un metodo mas facil de elaborar.

kmeans =  KMeans(n_clusters=cluster, random_state = 1)
kmeans.fit(nba[["ppg", "atr"]])
kmeans.labels_

nba["cluster"] = kmeans.labels_

visualize_cluster(nba, cluster)

nba.head(10)


