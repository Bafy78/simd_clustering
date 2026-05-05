import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import pyperf

def run_kmeans_lloyd(X, n_clusters, init_centers):
    with threadpool_limits(limits=1):
        kmeans = KMeans(
            n_clusters=n_clusters, 
            init=init_centers, 
            n_init=1, 
            max_iter=300, 
            algorithm="lloyd",
            tol=1e-4
        )
        kmeans.fit(X)
    return kmeans

def append_custom_args(cmd, args):
    cmd.extend(['--dataset-bin', args.dataset_bin])
    cmd.extend(['--n-samples', str(args.n_samples)])
    cmd.extend(['--n-features', str(args.n_features)])
    cmd.extend(['--n-clusters', str(args.n_clusters)])
    cmd.extend(['--init-centroids-bin', args.init_centroids_bin])
    cmd.extend(['--result-file', args.result_file])

if __name__ == "__main__":
    runner = pyperf.Runner(add_cmdline_args=append_custom_args, warmups=1, processes=1, values=20)
    runner.argparser.add_argument('--dataset-bin', required=True)
    runner.argparser.add_argument('--n-samples', type=int, required=True)
    runner.argparser.add_argument('--n-features', type=int, required=True)
    runner.argparser.add_argument('--n-clusters', type=int, required=True)
    runner.argparser.add_argument('--init-centroids-bin', required=True)
    runner.argparser.add_argument('--result-file', required=True)
    
    args = runner.parse_args()

    X = np.fromfile(args.dataset_bin, dtype=np.float32).reshape((args.n_samples, args.n_features))
    init_centers = np.fromfile(args.init_centroids_bin, dtype=np.float32).reshape((args.n_clusters, args.n_features))

    runner.bench_func('kmeans_lloyd_py', run_kmeans_lloyd, X, args.n_clusters, init_centers)

    # Worker check so only the master process writes the final output
    if not args.worker:
        final_kmeans = run_kmeans_lloyd(X, args.n_clusters, init_centers)
        centroids = final_kmeans.cluster_centers_
        labels = final_kmeans.labels_

        with open(args.result_file, 'w') as f:
            f.write("[Centroids]\n")
            for c in centroids:
                f.write(" ".join(f"{v:g}" for v in c) + "\n")
                
            f.write("[Clusters]\n")
            sets = {k: [] for k in range(args.n_clusters)}
            for i, label in enumerate(labels):
                sets[label].append(i)
                
            for k in range(args.n_clusters):
                f.write(f"{k}: " + " ".join(map(str, sets[k])) + "\n")