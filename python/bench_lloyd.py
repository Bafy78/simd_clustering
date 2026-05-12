import pyperf

threadpool_limits = None
np = None
KMeans = None
json = None


def import_runtime_deps():
    global threadpool_limits, KMeans, np

    from sklearn.cluster import KMeans as _KMeans
    from threadpoolctl import threadpool_limits as _threadpool_limits
    import numpy as _np

    KMeans = _KMeans
    threadpool_limits = _threadpool_limits
    np = _np


def run_kmeans_lloyd(X, n_clusters, init_centers):
    with threadpool_limits(limits=1):
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            max_iter=300,
            algorithm="lloyd",
            tol=1e-4,
        )
        kmeans.fit(X)
    return kmeans


def load_dataset(args):
    return np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.n_samples, args.n_features),
    )


def load_init_centers(args):
    return np.memmap(
        args.init_centroids_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.n_clusters, args.n_features),
    )


def compute_lloyd_metrics(X, labels, centroids, n_clusters, *, chunk_size=1_000_000):
    labels = np.asarray(labels, dtype=np.intp)
    centroids = np.asarray(centroids, dtype=np.float64)

    if n_clusters <= 0:
        raise RuntimeError("Invalid number of clusters")

    if centroids.shape[0] != n_clusters:
        raise RuntimeError("Centroid count does not match n_clusters")

    if labels.shape[0] != X.shape[0]:
        raise RuntimeError("Label count does not match point count")

    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise RuntimeError("Invalid cluster assignment")

    cluster_counts = np.bincount(labels, minlength=n_clusters).astype(np.int64)
    cluster_inertia = np.zeros(n_clusters, dtype=np.float64)

    n_samples = X.shape[0]

    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)

        X_chunk = np.asarray(X[start:stop], dtype=np.float64)
        labels_chunk = labels[start:stop]

        diff = X_chunk - centroids[labels_chunk]
        dist_sq = np.einsum("ij,ij->i", diff, diff)

        cluster_inertia += np.bincount(
            labels_chunk,
            weights=dist_sq,
            minlength=n_clusters,
        )

    total_inertia = float(cluster_inertia.sum())

    return {
        "inertia": total_inertia,
        "cluster_counts": cluster_counts.tolist(),
        "cluster_inertia": cluster_inertia.tolist(),
    }


def write_lloyd_metrics(path, *, X, kmeans, n_clusters):
    metrics = compute_lloyd_metrics(
        X,
        kmeans.labels_,
        kmeans.cluster_centers_,
        n_clusters,
    )

    payload = {
        "schema_version": 1,
        "language": "py",
        "iterations": int(kmeans.n_iter_),
        "inertia": metrics["inertia"],
        "cluster_counts": metrics["cluster_counts"],
        "cluster_inertia": metrics["cluster_inertia"],
        "centroids": np.asarray(kmeans.cluster_centers_, dtype=np.float64).tolist(),
        # Optional but useful for debugging sklearn drift.
        "sklearn_inertia": float(kmeans.inertia_),
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def append_custom_args(cmd, args):
    cmd.extend(["--dataset-bin", args.dataset_bin])
    cmd.extend(["--n-samples", str(args.n_samples)])
    cmd.extend(["--n-features", str(args.n_features)])
    cmd.extend(["--n-clusters", str(args.n_clusters)])
    cmd.extend(["--init-centroids-bin", args.init_centroids_bin])
    cmd.extend(["--metrics-file", args.metrics_file])


if __name__ == "__main__":
    runner = pyperf.Runner(
        add_cmdline_args=append_custom_args,
        warmups=1,
    )

    runner.argparser.add_argument("--dataset-bin", required=True)
    runner.argparser.add_argument("--n-samples", type=int, required=True)
    runner.argparser.add_argument("--n-features", type=int, required=True)
    runner.argparser.add_argument("--n-clusters", type=int, required=True)
    runner.argparser.add_argument("--init-centroids-bin", required=True)
    runner.argparser.add_argument("--metrics-file", required=True)

    args = runner.parse_args()

    if getattr(args, "worker", False):
        import_runtime_deps()

        X = load_dataset(args)
        init_centers = load_init_centers(args)
    else:
        X = None
        init_centers = None

    runner.bench_func(
        "kmeans_lloyd_py",
        run_kmeans_lloyd,
        X,
        args.n_clusters,
        init_centers,
    )

    if not getattr(args, "worker", False):
        import_runtime_deps()

        X = load_dataset(args)
        init_centers = load_init_centers(args)

        final_kmeans = run_kmeans_lloyd(X, args.n_clusters, init_centers)
        centroids = final_kmeans.cluster_centers_
        labels = final_kmeans.labels_
        iters = final_kmeans.n_iter_

        import json as _json

        json = _json

        write_lloyd_metrics(
            args.metrics_file,
            X=X,
            kmeans=final_kmeans,
            n_clusters=args.n_clusters,
        )
