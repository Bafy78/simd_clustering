from pathlib import Path

import pyperf

from benchmark_pipeline.gmm_covariance import SUPPORTED_GMM_COVARIANCE_TYPES

threadpool_limits = None
GaussianMixture = None
np = None
json = None


GMM_DEFAULT_TOL = 1e-3
GMM_DEFAULT_REG_COVAR = 1e-6
GMM_DEFAULT_MAX_ITER = 100
GMM_DEFAULT_N_INIT = 1


def import_runtime_deps():
    global threadpool_limits, GaussianMixture, np

    import numpy as _np
    from sklearn.mixture import GaussianMixture as _GaussianMixture
    from threadpoolctl import threadpool_limits as _threadpool_limits

    np = _np
    GaussianMixture = _GaussianMixture
    threadpool_limits = _threadpool_limits


def covariance_shape(covariance_type, K, D):
    if covariance_type == "full":
        return (K, D, D)
    if covariance_type == "diag":
        return (K, D)
    if covariance_type == "spherical":
        return (K,)
    raise RuntimeError(f"Unsupported covariance_type: {covariance_type!r}")


def load_dataset(args):
    return np.memmap(
        args.dataset_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.N, args.D),
    )


def load_gmm_weights(args):
    return np.memmap(
        args.gmm_weights_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.K,),
    )


def load_gmm_means(args):
    return np.memmap(
        args.gmm_means_bin,
        dtype=np.float32,
        mode="r",
        shape=(args.K, args.D),
    )


def load_gmm_precisions(args):
    return np.memmap(
        args.gmm_precisions_bin,
        dtype=np.float32,
        mode="r",
        shape=covariance_shape(args.covariance_type, args.K, args.D),
    )


def run_gmm_fit(X, K, covariance_type, weights, means, precisions):
    with threadpool_limits(limits=1):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type=covariance_type,
            tol=GMM_DEFAULT_TOL,
            reg_covar=GMM_DEFAULT_REG_COVAR,
            max_iter=GMM_DEFAULT_MAX_ITER,
            n_init=GMM_DEFAULT_N_INIT,
            weights_init=np.asarray(weights),
            means_init=np.asarray(means),
            precisions_init=np.asarray(precisions),
        )
        gmm.fit(X)

    return gmm


def write_gmm_metrics(path, *, gmm, covariance_type):
    payload = {
        "schema_version": 1,
        "phase": "gmm",
        "language": "py",
        "covariance_type": covariance_type,
        "algorithm_iterations": int(gmm.n_iter_),
        "lower_bound": float(gmm.lower_bound_),
        "lower_bounds": [float(value) for value in gmm.lower_bounds_],
        "weights": np.asarray(gmm.weights_, dtype=np.float64).tolist(),
        "means": np.asarray(gmm.means_, dtype=np.float64).tolist(),
        "covariances": np.asarray(gmm.covariances_, dtype=np.float64).tolist(),
        "sklearn_defaults": {
            "tol": GMM_DEFAULT_TOL,
            "reg_covar": GMM_DEFAULT_REG_COVAR,
            "max_iter": GMM_DEFAULT_MAX_ITER,
            "n_init": GMM_DEFAULT_N_INIT,
        },
    }

    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def append_custom_args(cmd, args):
    cmd.extend(["--dataset-bin", args.dataset_bin])
    cmd.extend(["--D", str(args.D)])
    cmd.extend(["--N", str(args.N)])
    cmd.extend(["--K", str(args.K)])
    cmd.extend(["--covariance-type", args.covariance_type])
    cmd.extend(["--gmm-weights-bin", args.gmm_weights_bin])
    cmd.extend(["--gmm-means-bin", args.gmm_means_bin])
    cmd.extend(["--gmm-precisions-bin", args.gmm_precisions_bin])
    cmd.extend(["--metrics-file", args.metrics_file])


if __name__ == "__main__":
    runner = pyperf.Runner(
        add_cmdline_args=append_custom_args,
        warmups=1,
    )

    runner.argparser.add_argument("--dataset-bin", required=True)
    runner.argparser.add_argument("--D", type=int, required=True)
    runner.argparser.add_argument("--N", type=int, required=True)
    runner.argparser.add_argument("--K", type=int, required=True)
    runner.argparser.add_argument(
        "--covariance-type",
        choices=SUPPORTED_GMM_COVARIANCE_TYPES,
        required=True,
    )
    runner.argparser.add_argument("--gmm-weights-bin", required=True)
    runner.argparser.add_argument("--gmm-means-bin", required=True)
    runner.argparser.add_argument("--gmm-precisions-bin", required=True)
    runner.argparser.add_argument("--metrics-file", required=True)

    args = runner.parse_args()

    if getattr(args, "worker", False):
        import_runtime_deps()

        X = load_dataset(args)
        weights = load_gmm_weights(args)
        means = load_gmm_means(args)
        precisions = load_gmm_precisions(args)
    else:
        X = None
        weights = None
        means = None
        precisions = None

    runner.bench_func(
        "gmm_em_py",
        run_gmm_fit,
        X,
        args.K,
        args.covariance_type,
        weights,
        means,
        precisions,
    )

    if not getattr(args, "worker", False):
        import_runtime_deps()

        X = load_dataset(args)
        weights = load_gmm_weights(args)
        means = load_gmm_means(args)
        precisions = load_gmm_precisions(args)

        final_gmm = run_gmm_fit(
            X,
            args.K,
            args.covariance_type,
            weights,
            means,
            precisions,
        )

        import json as _json

        json = _json

        write_gmm_metrics(
            args.metrics_file,
            gmm=final_gmm,
            covariance_type=args.covariance_type,
        )
