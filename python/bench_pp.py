import pyperf
import argparse
np = None
kmeans_plusplus = None

def run_kmeans_pp(X, n_clusters):
    centers, _ = kmeans_plusplus(X, n_clusters=n_clusters)
    return centers

def load_dataset(args):
    X = np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode='r',
        shape=(args.n_samples, args.n_features),
    )
    return X

def preparse_benchmark_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench-values", type=int, default=10)
    parser.add_argument("--bench-min-time", type=float, default=0.1)
    args, _ = parser.parse_known_args()

    if args.bench_values <= 0:
        raise SystemExit("--bench-values must be > 0")

    if args.bench_min_time <= 0:
        raise SystemExit("--bench-min-time must be > 0")

    return args

def append_custom_args(cmd, args):
    cmd.extend(['--dataset-bin', args.dataset_bin])
    cmd.extend(['--n-samples', str(args.n_samples)])
    cmd.extend(['--n-features', str(args.n_features)])
    cmd.extend(['--n-clusters', str(args.n_clusters)])
    cmd.extend(['--bench-values', str(args.bench_values)])
    cmd.extend(['--bench-min-time', str(args.bench_min_time)])

if __name__ == "__main__":
    bench_args = preparse_benchmark_args()
    runner = pyperf.Runner(
        add_cmdline_args=append_custom_args,
        warmups=1,
        processes=1,
        values=bench_args.bench_values,
        min_time=bench_args.bench_min_time,
    )
    runner.argparser.add_argument('--dataset-bin', required=True)
    runner.argparser.add_argument('--n-samples', type=int, required=True)
    runner.argparser.add_argument('--n-features', type=int, required=True)
    runner.argparser.add_argument('--n-clusters', type=int, required=True)
    runner.argparser.add_argument('--bench-values', type=int, default=bench_args.bench_values)
    runner.argparser.add_argument('--bench-min-time', type=float, default=bench_args.bench_min_time)
    args = runner.parse_args()

    if getattr(args, "worker", False):
        import numpy as _np
        from sklearn.cluster import kmeans_plusplus as _kmeans_plusplus
        np = _np
        kmeans_plusplus = _kmeans_plusplus

        X = load_dataset(args)
    else:
        X = None

    runner.bench_func('kmeans_pp_py', run_kmeans_pp, X, args.n_clusters)