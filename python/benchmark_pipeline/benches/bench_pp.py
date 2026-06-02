import pyperf

np = None
kmeans_plusplus = None


def import_runtime_deps():
    global np, kmeans_plusplus

    import numpy as _np
    from sklearn.cluster import kmeans_plusplus as _kmeans_plusplus

    np = _np
    kmeans_plusplus = _kmeans_plusplus


def run_kmeans_pp(X, K):
    centers, _ = kmeans_plusplus(X, n_clusters=K)
    return centers


def load_dataset(args):
    return np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.D),
    )


def append_custom_args(cmd, args):
    cmd.extend(["--dataset-bin", args.dataset_bin])
    cmd.extend(["--D", str(args.D)])
    cmd.extend(["--N", str(args.N)])
    cmd.extend(["--K", str(args.K)])


if __name__ == "__main__":
    runner = pyperf.Runner(
        add_cmdline_args=append_custom_args,
        warmups=1,
    )

    runner.argparser.add_argument("--dataset-bin", required=True)
    runner.argparser.add_argument("--D", type=int, required=True)
    runner.argparser.add_argument("--N", type=int, required=True)
    runner.argparser.add_argument("--K", type=int, required=True)

    args = runner.parse_args()

    if getattr(args, "worker", False):
        import_runtime_deps()
        X = load_dataset(args)
    else:
        X = None

    runner.bench_func(
        "kmeans_pp_py",
        run_kmeans_pp,
        X,
        args.K,
    )
