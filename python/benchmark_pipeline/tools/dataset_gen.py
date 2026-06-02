import argparse
import os
from typing import Literal

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

CovarianceType = Literal["full", "tied", "diag", "spherical"]


def nearest_center_labels(
    X: np.ndarray,
    centers: np.ndarray,
    *,
    max_distance_cells: int = 25_000_000,
) -> np.ndarray:
    """Assign each sample to its closest initial centroid without materializing the full N×K matrix."""
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
    *,
    reg_covar: float,
    max_chunk_rows: int = 1_000_000,
) -> np.ndarray:
    N, D = X.shape
    K = means.shape[0]
    sq_sums = np.zeros(K, dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, N, max_chunk_rows):
        stop = min(start + max_chunk_rows, N)
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
    *,
    reg_covar: float,
    max_chunk_rows: int = 1_000_000,
) -> np.ndarray:
    N, D = X.shape
    K = means.shape[0]
    sq_sums = np.zeros((K, D), dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, N, max_chunk_rows):
        stop = min(start + max_chunk_rows, N)
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
    *,
    reg_covar: float,
    max_chunk_rows: int = 250_000,
) -> np.ndarray:
    N, D = X.shape
    K = means.shape[0]
    covariances = np.zeros((K, D, D), dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, N, max_chunk_rows):
        stop = min(start + max_chunk_rows, N)
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


def _estimate_tied_precision(
    X: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    *,
    reg_covar: float,
    max_chunk_rows: int = 250_000,
) -> np.ndarray:
    N, D = X.shape
    covariance = np.zeros((D, D), dtype=np.float64)
    means64 = np.asarray(means, dtype=np.float64)

    for start in range(0, N, max_chunk_rows):
        stop = min(start + max_chunk_rows, N)
        labels_chunk = labels[start:stop]
        X_chunk = np.asarray(X[start:stop], dtype=np.float64)
        diff = X_chunk - means64[labels_chunk]
        covariance += diff.T @ diff

    covariance /= float(N)
    diag = np.arange(D)
    covariance[diag, diag] += reg_covar
    return np.linalg.inv(covariance)


def estimate_gmm_initial_parameters(
    X: np.ndarray,
    means: np.ndarray,
    *,
    covariance_type: CovarianceType,
    reg_covar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic GMM initialization artifacts from k-means++ means.

    The means are the k-means++ centers. We derive hard nearest-center assignments
    only to initialize mixture weights and covariance/precision parameters.
    """
    labels = nearest_center_labels(X, means)
    counts = np.bincount(labels, minlength=means.shape[0]).astype(np.int64)
    _check_non_empty_clusters(counts)

    weights = counts.astype(np.float64) / float(X.shape[0])

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
    elif covariance_type == "tied":
        precisions = _estimate_tied_precision(
            X,
            labels,
            means,
            reg_covar=reg_covar,
        )
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type!r}")

    return (
        weights.astype(np.float32),
        np.asarray(means, dtype=np.float32),
        np.asarray(precisions, dtype=np.float32),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--dataset-out", required=True)
    parser.add_argument("--centroids-out", required=True)
    parser.add_argument("--gmm-weights-out")
    parser.add_argument("--gmm-means-out")
    parser.add_argument("--gmm-precisions-out")
    parser.add_argument(
        "--gmm-covariance-type",
        choices=("full", "tied", "diag", "spherical"),
        default="spherical",
    )
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    return parser.parse_args()


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

    # 3. Generate GMM-specific initialization artifacts when requested.
    gmm_outputs = [args.gmm_weights_out, args.gmm_means_out, args.gmm_precisions_out]
    if any(gmm_outputs):
        if not all(gmm_outputs):
            raise RuntimeError(
                "Either provide all GMM init outputs "
                "(--gmm-weights-out, --gmm-means-out, --gmm-precisions-out), or none."
            )

        weights, means, precisions = estimate_gmm_initial_parameters(
            X_float32,
            centers_float32,
            covariance_type=args.gmm_covariance_type,
            reg_covar=args.gmm_reg_covar,
        )

        weights.tofile(args.gmm_weights_out)
        means.tofile(args.gmm_means_out)
        precisions.tofile(args.gmm_precisions_out)


if __name__ == "__main__":
    main()
