import sys
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

if len(sys.argv) < 5:
    print(f"Usage: python {sys.argv[0]} <binary_file> <n_samples> <n_features> <n_clusters>")
    sys.exit(1)

filename = sys.argv[1]
n_samples = int(sys.argv[2])
n_features = int(sys.argv[3])
n_clusters = int(sys.argv[4])

# Load the contiguous float32 block and reshape it into an (N, D) array
raw_data = np.fromfile(filename, dtype=np.float32)

if raw_data.size != n_samples * n_features:
    print(f"Error: Expected {n_samples * n_features} floats, but read {raw_data.size}.")
    sys.exit(1)

X = raw_data.reshape((n_samples, n_features))

with threadpool_limits(limits=1):
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init="k-means++", 
        n_init=1, 
        max_iter=100, 
        algorithm="lloyd",
        tol=1e-4,
        random_state=42 # Fixed seed to ensure initialization is identical across benchmark runs
    )
    kmeans.fit(X)

# Only printing the final centroids so standard output doesn't skew the benchmark time
centroids = kmeans.cluster_centers_

print("Final Centroids:")
for k in range(n_clusters):
    c_str = ", ".join(f"{v:g}" for v in centroids[k])
    print(f"Cluster {k}: ({c_str})")