import pyperf
import argparse

threadpool_limits = None
memmap = None
float32 = None
KMeans = None


def import_runtime_deps():
    global threadpool_limits, memmap, float32, KMeans

    from sklearn.cluster import KMeans as _KMeans
    from threadpoolctl import threadpool_limits as _threadpool_limits
    from numpy import memmap as _memmap
    from numpy import float32 as _float32

    KMeans = _KMeans
    threadpool_limits = _threadpool_limits
    memmap = _memmap
    float32 = _float32


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
    return memmap(
        args.dataset_bin,
        dtype=float32,
        mode="r",
        shape=(args.n_samples, args.n_features),
    )


def load_init_centers(args):
    return memmap(
        args.init_centroids_bin,
        dtype=float32,
        mode="r",
        shape=(args.n_clusters, args.n_features),
    )


def append_custom_args(cmd, args):
    cmd.extend(["--dataset-bin", args.dataset_bin])
    cmd.extend(["--n-samples", str(args.n_samples)])
    cmd.extend(["--n-features", str(args.n_features)])
    cmd.extend(["--n-clusters", str(args.n_clusters)])
    cmd.extend(["--init-centroids-bin", args.init_centroids_bin])
    cmd.extend(["--result-file", args.result_file])


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
    runner.argparser.add_argument("--result-file", required=True)

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

        with open(args.result_file, "w") as f:
            f.write("[Lloyd Iterations]\n")
            f.write(f"{iters}\n")

            f.write("[Centroids]\n")
            for c in centroids:
                f.write(" ".join(f"{v:g}" for v in c) + "\n")

            f.write("[Clusters]\n")
            sets = {k: [] for k in range(args.n_clusters)}
            for i, label in enumerate(labels):
                sets[label].append(i)

            for k in range(args.n_clusters):
                f.write(f"{k}: " + " ".join(map(str, sets[k])) + "\n")
