import numpy as np
from sklearn.cluster import kmeans_plusplus
import pyperf

def run_kmeans_pp(X, n_clusters):
    centers, _ = kmeans_plusplus(X, n_clusters=n_clusters)
    return centers

def append_custom_args(cmd, args):
    cmd.extend(['--dataset-bin', args.dataset_bin])
    cmd.extend(['--n-samples', str(args.n_samples)])
    cmd.extend(['--n-features', str(args.n_features)])
    cmd.extend(['--n-clusters', str(args.n_clusters)])

if __name__ == "__main__":
    runner = pyperf.Runner(add_cmdline_args=append_custom_args, warmups=1, processes=1, values=20)
    runner.argparser.add_argument('--dataset-bin', required=True)
    runner.argparser.add_argument('--n-samples', type=int, required=True)
    runner.argparser.add_argument('--n-features', type=int, required=True)
    runner.argparser.add_argument('--n-clusters', type=int, required=True)
    args = runner.parse_args()

    raw_data = np.fromfile(args.dataset_bin, dtype=np.float32)
    X = raw_data.reshape((args.n_samples, args.n_features))

    runner.bench_func('kmeans_pp_py', run_kmeans_pp, X, args.n_clusters)