import xlrd
import xlwt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Select the Optimal K Value (Elbow Method)
def find_optimal_k(data, max_k=5):
    distortions = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plot the elbow method
    plt.figure(figsize=(12, 6))
    plt.plot(range(2, max_k + 1), distortions, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

    # Plot the silhouette scores
    plt.figure(figsize=(12, 6))
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Return the k value with the highest silhouette score
    optimal_k = np.argmax(silhouette_scores) + 2
    return optimal_k


# data = np.genfromtxt('./Data_.txt') #if input is a text file
# data = pd.read_excel('cation.xlsx', header=None) #if input is an excel file
# Load data from a CSV file
data = pd.read_csv('./Data_.csv').values

# data = data.drop(columns=['Unnamed: 0'])
data = data.iloc[:, 1:10]  # Select the first 9 columns
data = data[1:]  # Remove the first row from the DataFrame


def KMeans_N(KK):
    model = KMeans(n_clusters=KK)
    model.fit(data)

    centers = model.cluster_centers_

    result = model.predict(data)
    # 利用calinski_harabasz
    CH_index = metrics.calinski_harabasz_score(data, result)
    return CH_index


CH_index_result = []
plt.xlabel("N_Cluster")
plt.ylabel("calinski_harabasz_score")
plt.title('Chose The Right Cluster Number')

for ii in range(2, 10):
    temp_list = []
    temp = KMeans_N(ii)
    temp_list.append(ii)
    temp_list.append(temp)
    CH_index_result.append(temp_list)
    plt.scatter(ii, temp)

# plt.show()

k = 4

model = KMeans(n_clusters=k)
model.fit(data)

centers = model.cluster_centers_

result = model.predict(data)
np.savetxt('./Kmeans_result.txt', result)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load your data
# data = np.genfromtxt('./Data_.txt')  # Uncomment this line if you have data in a text file
# data = pd.read_excel('./cation.xlsx', header=None).iloc[:, 1:10]  # Modify this as per your actual data

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)


def calculate_metrics(KK):
    model = KMeans(n_clusters=KK, random_state=42)
    model.fit(data)

    centers = model.cluster_centers_
    result = model.predict(data)

    # Calculate Calinski-Harabasz index
    CH_index = metrics.calinski_harabasz_score(data, result)

    # Calculate Silhouette Score
    if KK > 1:  # Silhouette Score requires at least 2 clusters
        silhouette_avg = silhouette_score(data, result)
    else:
        silhouette_avg = -1  # Invalid score for 1 cluster

    return CH_index, silhouette_avg


# Elbow Method for WCSS
wcss = []
for ii in range(1, 11):
    model = KMeans(n_clusters=ii, random_state=42)
    model.fit(data)
    wcss.append(model.inertia_)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")

# CH Index and Silhouette Score for different k
CH_index_results = []
silhouette_scores = []
for ii in range(2, 11):  # k needs to be at least 2 for silhouette score
    CH_index, silhouette_avg = calculate_metrics(ii)
    CH_index_results.append((ii, CH_index))
    silhouette_scores.append((ii, silhouette_avg))

# Plot Calinski-Harabasz Index
plt.subplot(1, 2, 2)
CH_index_results = np.array(CH_index_results)
plt.plot(CH_index_results[:, 0], CH_index_results[:, 1], marker='o', label='CH Index')
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Score")
plt.title("Calinski-Harabasz Index for Optimal k")

# Plot Silhouette Score
plt.subplot(1, 2, 2)
silhouette_scores = np.array(silhouette_scores)
plt.plot(silhouette_scores[:, 0], silhouette_scores[:, 1], marker='o', label='Silhouette Score')
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.title("Silhouette Score for Optimal k")
plt.legend()

plt.tight_layout()
plt.show()

# Choose optimal k based on the plots
optimal_k = 4  # Change this based on the plots
model = KMeans(n_clusters=optimal_k, random_state=42)
model.fit(data)

result = model.predict(data)
np.savetxt('./Kmeans_result.csv', result)
