import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [1.0, 1.0], [2.0, 1.5], [3.0, 2.0], [4.0, 2.5],
    [5.0, 3.0], [6.0, 3.5], [7.0, 4.0], [8.0, 4.5]
], dtype=np.float32)

initial_centroids = np.array([
    [2.5, 3.0],
    [6.0, 2.0]
], dtype=np.float32)

kmeans = KMeans(
    n_clusters=2, 
    init=initial_centroids, 
    n_init=1, 
    max_iter=100, 
    algorithm="lloyd"
)

kmeans.fit(X)

print(f"{'Point (x, y)':<18} {'Cluster ID':<12} {'Centroid (cx, cy)'}")
print("-" * 55)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(len(X)):
    cluster_id = labels[i]
    point = X[i]
    centroid = centroids[cluster_id]
    
    print(f"({point[0]:>4.1f}, {point[1]:>4.1f}) {cluster_id:>8}            ({centroid[0]:.4f}, {centroid[1]:.4f})")