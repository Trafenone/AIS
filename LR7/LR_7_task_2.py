from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Завантаження набору даних Iris
iris = load_iris()
data = iris['data']
labels_true = iris['target']

# Використання алгоритму KMeans для 5 кластерів
kmeans_model = KMeans(n_clusters=5, random_state=42)
kmeans_model.fit(data)
predicted_clusters = kmeans_model.predict(data)

# Візуалізація результатів кластеризації
plt.scatter(data[:, 0], data[:, 1], c=predicted_clusters, s=50, cmap='viridis')
cluster_centers = kmeans_model.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("Кластеризація KMeans з 5 кластерами")
plt.show()

# Визначення власної реалізації алгоритму кластеризації
def custom_clustering(data, n_clusters, seed=2):
    rng = np.random.RandomState(seed)
    # Ініціалізація центрів кластерів випадковим вибором точок
    initial_centers = data[rng.permutation(data.shape[0])[:n_clusters]]
    centers = initial_centers

    while True:
        # Призначення точок до найближчих центрів
        cluster_assignments = pairwise_distances_argmin(data, centers)
        # Оновлення центрів кластерів як середнє точок у кожному кластері
        updated_centers = np.array([data[cluster_assignments == i].mean(0) for i in range(n_clusters)])
        # Якщо центри не змінюються, завершити цикл
        if np.all(centers == updated_centers):
            break
        centers = updated_centers

    return centers, cluster_assignments

# Використання власної реалізації кластеризації з різними seed
centers, cluster_labels = custom_clustering(data, 3)
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=50, cmap='viridis')
plt.title("Кастомна кластеризація (seed=2)")
plt.show()

centers, cluster_labels = custom_clustering(data, 3, seed=0)
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=50, cmap='viridis')
plt.title("Кастомна кластеризація (seed=0)")
plt.show()

# Порівняння результатів з KMeans для 3 кластерів
kmeans_labels = KMeans(n_clusters=3, random_state=0).fit_predict(data)
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels, s=50, cmap='viridis')
plt.title("Кластеризація KMeans з 3 кластерами")
plt.show()
