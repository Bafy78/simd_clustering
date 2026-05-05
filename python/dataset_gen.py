import argparse
import os
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import kmeans_plusplus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--n-features', type=int, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    parser.add_argument('--dataset-out', required=True)
    parser.add_argument('--centroids-out', required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dataset_out), exist_ok=True)

    # 1. Generate Dataset
    X, _, *_ = make_blobs(n_samples=args.n_samples, n_features=args.n_features, centers=args.n_clusters)
    X_float32 = X.astype(np.float32)
    X_float32.tofile(args.dataset_out)

    # 2. Generate Initial Centroids
    centers, _ = kmeans_plusplus(X_float32, n_clusters=args.n_clusters)
    centers_float32 = centers.astype(np.float32)
    centers_float32.tofile(args.centroids_out)