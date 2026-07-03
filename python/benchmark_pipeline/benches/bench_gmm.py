from pathlib import Path

import pyperf

from benchmark_pipeline.gmm_covariance import SUPPORTED_GMM_COVARIANCE_TYPES

threadpool_limits = None
GaussianMixture = None
FitOnlyGaussianMixture = None
np = None
json = None


GMM_DEFAULT_TOL = 1e-3
GMM_DEFAULT_REG_COVAR = 1e-6
GMM_DEFAULT_MAX_ITER = 100
GMM_DEFAULT_N_INIT = 1


def import_runtime_deps():
    global threadpool_limits, GaussianMixture, FitOnlyGaussianMixture, np

    import warnings as _warnings

    import numpy as _np
    import sklearn as _sklearn
    from sklearn.base import _fit_context
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture as _GaussianMixture
    from sklearn.utils import check_random_state
    from sklearn.utils._array_api import get_namespace
    from sklearn.utils.validation import validate_data
    from threadpoolctl import threadpool_limits as _threadpool_limits

    if _sklearn.__version__ != "1.8.0":
        raise RuntimeError(
            "FitOnlyGaussianMixture copies sklearn.mixture.BaseMixture.fit_predict "
            "control flow and is only audited for scikit-learn==1.8.0."
        )

    class _FitOnlyGaussianMixture(_GaussianMixture):
        @_fit_context(prefer_skip_nested_validation=True)
        def fit(self, X, y=None):
            """
            scikit-learn 1.8.0 GaussianMixture.fit_predict(), copied for benchmarking,
            except that the final fit_predict-only E-step + argmax tail is removed.

            This keeps:
              - sklearn input validation
              - parameter validation
              - initialization
              - EM loop ordering
              - convergence warning behavior
              - final parameter/state assignment

            This removes:
              - final _e_step(X)
              - argmax(log_resp)
            """
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                ensure_min_samples=2,
            )

            if X.shape[0] < self.n_components:
                raise ValueError(
                    "Expected n_samples >= n_components "
                    f"but got n_components = {self.n_components}, "
                    f"n_samples = {X.shape[0]}"
                )

            self._check_parameters(X, xp=xp)

            # Same warm_start / n_init handling as sklearn 1.8.0.
            do_init = not (self.warm_start and hasattr(self, "converged_"))
            n_init = self.n_init if do_init else 1

            max_lower_bound = -xp.inf
            best_lower_bounds = []
            self.converged_ = False

            random_state = check_random_state(self.random_state)

            for init in range(n_init):
                self._print_verbose_msg_init_beg(init)

                if do_init:
                    self._initialize_parameters(X, random_state, xp=xp)

                lower_bound = -xp.inf if do_init else self.lower_bound_
                current_lower_bounds = []

                if self.max_iter == 0:
                    best_params = self._get_parameters()
                    best_n_iter = 0
                else:
                    converged = False

                    for n_iter in range(1, self.max_iter + 1):
                        prev_lower_bound = lower_bound

                        log_prob_norm, log_resp = self._e_step(X, xp=xp)
                        self._m_step(X, log_resp, xp=xp)
                        lower_bound = self._compute_lower_bound(
                            log_resp,
                            log_prob_norm,
                        )
                        current_lower_bounds.append(lower_bound)

                        change = lower_bound - prev_lower_bound
                        self._print_verbose_msg_iter_end(n_iter, change)

                        if abs(change) < self.tol:
                            converged = True
                            break

                    self._print_verbose_msg_init_end(lower_bound, converged)

                    if lower_bound > max_lower_bound or max_lower_bound == -xp.inf:
                        max_lower_bound = lower_bound
                        best_params = self._get_parameters()
                        best_n_iter = n_iter
                        best_lower_bounds = current_lower_bounds
                        self.converged_ = converged

            if not self.converged_ and self.max_iter > 0:
                _warnings.warn(
                    (
                        "Best performing initialization did not converge. "
                        "Try different init parameters, or increase max_iter, "
                        "tol, or check for degenerate data."
                    ),
                    ConvergenceWarning,
                )

            self._set_parameters(best_params, xp=xp)
            self.n_iter_ = best_n_iter
            self.lower_bound_ = max_lower_bound
            self.lower_bounds_ = best_lower_bounds

            return self

    np = _np
    GaussianMixture = _GaussianMixture
    FitOnlyGaussianMixture = _FitOnlyGaussianMixture
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
    gmm = FitOnlyGaussianMixture(
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

    def bench():
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

    if getattr(args, "worker", False):
        with threadpool_limits(limits=1):
            bench()
    else:
        bench()

    if not getattr(args, "worker", False):
        import_runtime_deps()

        X = load_dataset(args)
        weights = load_gmm_weights(args)
        means = load_gmm_means(args)
        precisions = load_gmm_precisions(args)

        with threadpool_limits(limits=1):
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
