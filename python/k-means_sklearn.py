import sys
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import pyperf

def run_clustering(X, n_clusters):
    with threadpool_limits(limits=1):
        kmeans = KMeans(
            n_clusters=n_clusters, 
            init="k-means++", 
            n_init=1, 
            max_iter=300, 
            algorithm="lloyd",
            tol=1e-4
        )
        kmeans.fit(X)
    return kmeans

def append_custom_args(cmd, args):
    cmd.extend(['--binary-file', args.binary_file])
    cmd.extend(['--n-samples', str(args.n_samples)])
    cmd.extend(['--n-features', str(args.n_features)])
    cmd.extend(['--n-clusters', str(args.n_clusters)])
    cmd.extend(['--result-file', args.result_file])

if __name__ == "__main__":
    runner = pyperf.Runner(add_cmdline_args=append_custom_args, warmups=1, processes=1, values=20)
    runner.argparser.add_argument('--binary-file', required=True)
    runner.argparser.add_argument('--n-samples', type=int, required=True)
    runner.argparser.add_argument('--n-features', type=int, required=True)
    runner.argparser.add_argument('--n-clusters', type=int, required=True)
    runner.argparser.add_argument('--result-file', required=True)
    
    args = runner.parse_args()

    # Load the contiguous float32 block and reshape it into an (N, D) array
    raw_data = np.fromfile(args.binary_file, dtype=np.float32)

    if raw_data.size != args.n_samples * args.n_features:
        print(f"Error: Expected {args.n_samples * args.n_features} floats, but read {raw_data.size}.")
        sys.exit(1)

    X = raw_data.reshape((args.n_samples, args.n_features))

    runner.bench_func('kmeans_fit', run_clustering, X, args.n_clusters)

    # 2. Write the verification results
    # pyperf sets args.worker to True in its spawned processes. 
    # We check this so only the master process writes the final output file.
    if not args.worker:
        final_kmeans = run_clustering(X, args.n_clusters)
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