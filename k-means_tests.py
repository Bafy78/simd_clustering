import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    # Cluster 1 roughly around (1, 1, 1)
    [1.0, 1.0, 1.5], [1.2, 1.1, 1.4], [0.8, 1.3, 1.6],
    [1.1, 0.9, 1.2], [1.5, 1.2, 1.7], [0.9, 1.0, 1.5],
    [1.3, 1.4, 1.3], [1.0, 1.2, 1.1], [1.4, 0.8, 1.4],
    [1.2, 1.5, 1.6],
    
    # Cluster 2 roughly around (5, 5, 5)
    [5.0, 5.0, 5.0], [5.2, 4.8, 5.1], [4.9, 5.3, 4.8],
    [5.1, 5.1, 5.2], [4.8, 4.9, 4.9], [5.3, 5.0, 5.3],
    [5.0, 5.2, 4.7], [4.7, 5.1, 5.0], [5.2, 5.3, 5.1],
    [4.9, 4.7, 5.2],
    
    # Cluster 3 roughly around (8, 1, 8)
    [8.0, 1.0, 8.0], [8.1, 1.2, 7.8], [7.9, 0.9, 8.2],
    [8.2, 1.1, 8.1], [7.8, 1.0, 7.9], [8.3, 0.8, 8.0],
    [8.0, 1.3, 7.7], [7.7, 1.1, 8.3], [8.1, 0.9, 8.1],
    [7.9, 1.2, 7.8]
], dtype=np.float32)

initial_centroids = np.array([
    [2.5, 3.0, 1.0], 
    [6.0, 2.0, 4.5],
    [7.5, 4.0, 7.0]
], dtype=np.float32)

kmeans = KMeans(
    n_clusters=initial_centroids.shape[0], 
    init=initial_centroids, 
    n_init=1, 
    max_iter=100, 
    algorithm="lloyd",
    tol=1e-4
)

kmeans.fit(X)

num_dims = X.shape[1]
coord_width = num_dims * 8

# Print dynamic headers
print(f"{'Point':<{coord_width}} {'Cluster ID':<12} Centroid")
print("-" * (coord_width + 12 + coord_width))

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(len(X)):
    cluster_id = labels[i]
    point = X[i]
    centroid = centroids[cluster_id]
    
    # Dynamically format coordinates for arbitrary dimensions
    pt_str = "(" + ", ".join(f"{v:>5.1f}" for v in point) + ")"
    c_str = "(" + ", ".join(f"{v:>5.5f}" for v in centroid) + ")"
    
    print(f"{pt_str:<{coord_width}} {cluster_id:>8}      {c_str}")