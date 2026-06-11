import argparse
import os
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parents[2]
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

from benchmark_pipeline.gmm_covariance import validate_gmm_covariance_type


def nearest_center_labels(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    max_distance_cells: int = 25_000_000,
) -> np.ndarray:
    """Assign each sample to its closest initial centroid without materializing the full sample-by-cluster matrix."""
    N, D = X.shape
    K = centers.shape[0]
    chunk_size = max(1, max_distance_cells // max(1, K * D))

    labels = np.empty(N, dtype=np.intp)

    centers64 = np.asarray(centers, dtype=np.float64)
    center_norms = np.einsum("kd,kd->k", centers64, centers64)

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        X_chunk = np.asarray(X[start:stop], dtype=np.float64)

        # ||x - c||² = ||x||² - 2 x·c + ||c||². The ||x||² term is irrelevant
        # for argmin, so we avoid storing a 3-D differences tensor.
        scores = X_chunk @ centers64.T
        scores *= -2.0
        scores += center_norms
        labels[start:stop] = np.argmin(scores, axis=1)

    return labels


def _check_non_empty_clusters(counts: np.ndarray) -> None:
    empty = np.flatnonzero(counts == 0)
    if empty.size:
        preview = ", ".join(str(int(k)) for k in empty[:10])
        raise RuntimeError(
            "GMM initialization produced empty clusters. "
            f"First empty cluster ids: {preview}"
        )


def _estimate_spherical_precisions(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    reg_covar: float,
    max_chunk_rows: int = 1_000_000,
) -> np.ndarray:
    _, D = X.shape
    K = means.shape[0]
    sq_sums = np.zeros(K, dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, X.shape[0], max_chunk_rows):
        stop = min(start + max_chunk_rows, X.shape[0])
        labels_chunk = labels[start:stop]
        X_chunk = np.asarray(X[start:stop], dtype=np.float64)
        diff = X_chunk - means64[labels_chunk]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        sq_sums += np.bincount(labels_chunk, weights=dist_sq, minlength=K)

    covariances = sq_sums / (counts.astype(np.float64) * float(D))
    covariances += reg_covar
    return 1.0 / covariances


def _estimate_diag_precisions(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    reg_covar: float,
    max_chunk_rows: int = 1_000_000,
) -> np.ndarray:
    _, D = X.shape
    K = means.shape[0]
    sq_sums = np.zeros((K, D), dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, X.shape[0], max_chunk_rows):
        stop = min(start + max_chunk_rows, X.shape[0])
        labels_chunk = labels[start:stop]
        X_chunk = np.asarray(X[start:stop], dtype=np.float64)
        diff_sq = (X_chunk - means64[labels_chunk]) ** 2

        for d in range(D):
            sq_sums[:, d] += np.bincount(
                labels_chunk,
                weights=diff_sq[:, d],
                minlength=K,
            )

    covariances = sq_sums / counts.astype(np.float64)[:, np.newaxis]
    covariances += reg_covar
    return 1.0 / covariances


def _estimate_full_precisions(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    reg_covar: float,
    max_chunk_rows: int = 250_000,
) -> np.ndarray:
    _, D = X.shape
    K = means.shape[0]
    covariances = np.zeros((K, D, D), dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, X.shape[0], max_chunk_rows):
        stop = min(start + max_chunk_rows, X.shape[0])
        labels_chunk = labels[start:stop]
        X_chunk = np.asarray(X[start:stop], dtype=np.float64)

        present = np.unique(labels_chunk)
        for cluster in present:
            mask = labels_chunk == cluster
            diff = X_chunk[mask] - means64[cluster]
            covariances[cluster] += diff.T @ diff

    covariances /= counts.astype(np.float64)[:, np.newaxis, np.newaxis]
    diag = np.arange(D)
    covariances[:, diag, diag] += reg_covar
    return np.linalg.inv(covariances)


def estimate_gmm_shared_initial_parameters(
    X: np.ndarray,
    means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return shared GMM initialization state used by every covariance type."""
    labels = nearest_center_labels(X, means)
    counts = np.bincount(labels, minlength=means.shape[0]).astype(np.int64)
    _check_non_empty_clusters(counts)

    weights = counts.astype(np.float64) / float(X.shape[0])

    return (
        weights.astype(np.float32),
        np.asarray(means, dtype=np.float32),
        labels,
        counts,
    )


def estimate_gmm_precisions(
    X: np.ndarray,
    means: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    covariance_type: str,
    reg_covar: float,
) -> np.ndarray:
    if covariance_type == "spherical":
        precisions = _estimate_spherical_precisions(
            X,
            labels,
            means,
            counts,
            reg_covar=reg_covar,
        )
    elif covariance_type == "diag":
        precisions = _estimate_diag_precisions(
            X,
            labels,
            means,
            counts,
            reg_covar=reg_covar,
        )
    elif covariance_type == "full":
        precisions = _estimate_full_precisions(
            X,
            labels,
            means,
            counts,
            reg_covar=reg_covar,
        )
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type!r}")

    return np.asarray(precisions, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dataset-out", required=True)
    parser.add_argument("--centroids-out", required=True)
    parser.add_argument("--gmm-weights-out")
    parser.add_argument("--gmm-means-out")
    parser.add_argument(
        "--gmm-precisions-out",
        nargs=2,
        action="append",
        metavar=("COVARIANCE_TYPE", "PATH"),
        default=[],
        help=(
            "Request a GMM precision init file for one covariance type. "
            "May be repeated, for example: --gmm-precisions-out full path.bin."
        ),
    )
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    return parser.parse_args()


def _parse_gmm_precision_outputs(raw_outputs: list[list[str]]) -> dict[str, str]:
    outputs: dict[str, str] = {}

    for covariance_type, path in raw_outputs:
        validate_gmm_covariance_type(covariance_type)

        if covariance_type in outputs:
            raise RuntimeError(
                f"Duplicate precision output for GMM covariance type {covariance_type!r}"
            )

        outputs[covariance_type] = path

    return outputs


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.dataset_out), exist_ok=True)

    # 1. Generate Dataset
    X, _, *_ = make_blobs(
        n_samples=args.N,
        n_features=args.D,
        centers=args.K,
        random_state=42,
    )
    X_float32 = X.astype(np.float32)
    X_float32.tofile(args.dataset_out)

    # 2. Generate Initial Centroids
    centers, _ = kmeans_plusplus(X_float32, n_clusters=args.K, random_state=42)
    centers_float32 = centers.astype(np.float32)
    centers_float32.tofile(args.centroids_out)

    # 3. Generate GMM-specific initialization artifacts only when requested.
    precision_outputs = _parse_gmm_precision_outputs(args.gmm_precisions_out)
    has_gmm_outputs = bool(args.gmm_weights_out or args.gmm_means_out or precision_outputs)

    if not has_gmm_outputs:
        return

    if not args.gmm_weights_out or not args.gmm_means_out or not precision_outputs:
        raise RuntimeError(
            "GMM initialization requires --gmm-weights-out, --gmm-means-out, "
            "and at least one repeated --gmm-precisions-out COVARIANCE_TYPE PATH."
        )

    weights, means, labels, counts = estimate_gmm_shared_initial_parameters(
        X_float32,
        centers_float32,
    )

    weights.tofile(args.gmm_weights_out)
    means.tofile(args.gmm_means_out)

    for covariance_type, path in precision_outputs.items():
        precisions = estimate_gmm_precisions(
            X_float32,
            means,
            labels,
            counts,
            covariance_type=covariance_type,
            reg_covar=args.gmm_reg_covar,
        )
        precisions.tofile(path)


if __name__ == "__main__":
    main()
