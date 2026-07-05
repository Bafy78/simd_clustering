import argparse
import hashlib
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

from benchmark_pipeline.gmm_covariance import validate_gmm_covariance_type
from benchmark_pipeline.tasks import validate_dataset_key


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
    covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    diag = np.arange(D)
    covariances[:, diag, diag] += reg_covar
    precisions = np.linalg.inv(covariances)
    return 0.5 * (precisions + np.swapaxes(precisions, 1, 2))


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
    parser.add_argument("--dataset", default="blobs")
    parser.add_argument(
        "--source-kind",
        choices=(
            "blobs",
            "local",
            "url",
            "openml",
            "uci",
            "huggingface",
        ),
        default="blobs",
        help="Input source provider. 'blobs' keeps the current synthetic data path.",
    )
    parser.add_argument(
        "--source-path",
        help="Path for local curated datasets.",
    )
    parser.add_argument(
        "--source-url",
        help="Direct dataset URL for --source-kind url.",
    )
    parser.add_argument(
        "--source-format",
        choices=("npy", "binary_f32"),
        help="Dense input format for local/url sources.",
    )
    parser.add_argument(
        "--downloads-dir",
        default="downloads",
        help="Directory used by remote fetchers and direct URL downloads.",
    )
    parser.add_argument("--openml-data-id", type=int)
    parser.add_argument("--openml-name")
    parser.add_argument("--openml-version")
    parser.add_argument("--uci-dataset-id", type=int)
    parser.add_argument("--hf-repo")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-feature-column")
    parser.add_argument(
        "--feature-columns",
        nargs="+",
        default=[],
        help="Scalar numeric columns to stack as features when a source supports columns.",
    )
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


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


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


def _source_format_for_args(args: argparse.Namespace) -> str | None:
    if args.source_format is not None:
        return args.source_format
    if args.source_url:
        suffix = Path(urlparse(args.source_url).path).suffix.lower()
        if suffix in {".npy", ".npz"}:
            return "npy"
    return None


def _safe_download_name(url: str, dataset: str) -> str:
    parsed_name = Path(urlparse(url).path).name
    parsed_name = re.sub(r"[^A-Za-z0-9._-]+", "_", parsed_name).strip("._")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    if parsed_name:
        return f"{dataset}_{digest}_{parsed_name}"
    return f"{dataset}_{digest}.download"


def _download_url(url: str, downloads_dir: str | Path, dataset: str) -> Path:
    downloads_dir = Path(downloads_dir).expanduser()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    destination = downloads_dir / _safe_download_name(url, dataset)
    if destination.exists():
        return destination

    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_destination)
        tmp_destination.replace(destination)
    finally:
        if tmp_destination.exists():
            tmp_destination.unlink()

    return destination


def _load_blob_dataset(N: int, D: int, K: int) -> np.ndarray:
    """Return the center-box half-width used for synthetic blob generation.

    The scaling reduces the default make_blobs artifact where random center
    distances grow with sqrt(D), while higher K makes centers denser in a
    fixed box. The reference point D=10, K=10 preserves the old B=10 scale.
    """
    reference_D = 10
    reference_K = 10
    reference_B = 10.0

    B = (
        reference_B
        * np.sqrt(reference_D / D)
        * (K / reference_K) ** (1.0 / D)
    )

    X, _, *_ = make_blobs(
        n_samples=N,
        n_features=D,
        centers=K,
        cluster_std=1.0,
        center_box=(-B, B),
        random_state=42,
    )

    return X


def _load_npy_dataset(path: str | Path) -> np.ndarray:
    data = np.load(Path(path).expanduser(), allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            keys = list(data.files)
            if len(keys) != 1:
                raise ValueError(
                    "NPZ real dataset inputs must contain exactly one array; "
                    f"found keys {keys!r}."
                )
            return np.asarray(data[keys[0]])
        finally:
            data.close()
    return np.asarray(data)


def _load_binary_f32_dataset(path: str | Path, N: int, D: int) -> np.ndarray:
    values = np.fromfile(Path(path).expanduser(), dtype=np.float32)
    expected_size = int(N) * int(D)
    if values.size != expected_size:
        raise ValueError(
            f"binary_f32 dataset has {values.size} float32 values, "
            f"expected {expected_size} for shape ({N}, {D})."
        )
    return values.reshape((N, D))


def _load_local_dataset(
    path: str | Path,
    source_format: str,
    N: int,
    D: int,
) -> np.ndarray:
    if source_format == "npy":
        return _load_npy_dataset(path)
    if source_format == "binary_f32":
        return _load_binary_f32_dataset(path, N, D)
    raise ValueError(f"Unsupported local dataset format: {source_format!r}")


def _coerce_columnar_matrix(source: object, columns: list[str]) -> np.ndarray:
    if not columns:
        raise ValueError("At least one feature column is required.")

    values: list[np.ndarray] = []
    for column in columns:
        column_values = np.asarray(source[column])  # type: ignore[index]
        if column_values.ndim != 1:
            raise ValueError(
                f"Feature column {column!r} must be scalar-valued, "
                f"got shape {column_values.shape!r}."
            )
        values.append(column_values)

    return np.column_stack(values)


def _coerce_vector_column(source: object, column: str) -> np.ndarray:
    column_values = source[column]  # type: ignore[index]
    array = np.asarray(column_values)

    # Some column stores expose list/array values as dtype=object. Materialize
    # them as rows so validation can enforce dense numeric 2-D data.
    if array.dtype == object:
        array = np.asarray(list(column_values))

    return array


def _load_openml_dataset(args: argparse.Namespace) -> np.ndarray:
    try:
        from sklearn.datasets import fetch_openml
    except ImportError as exc:
        raise RuntimeError(
            "OpenML dataset sources require scikit-learn with fetch_openml available."
        ) from exc

    if args.openml_data_id is None and not args.openml_name:
        raise ValueError("OpenML sources require --openml-data-id or --openml-name.")

    kwargs: dict[str, object] = {
        "as_frame": False,
        "return_X_y": False,
        "data_home": str(Path(args.downloads_dir).expanduser() / "openml"),
    }
    if args.openml_data_id is not None:
        kwargs["data_id"] = args.openml_data_id
    else:
        kwargs["name"] = args.openml_name
        if args.openml_version is not None:
            version: int | str = args.openml_version
            if isinstance(version, str) and version.isdigit():
                version = int(version)
            kwargs["version"] = version

    bunch = fetch_openml(**kwargs)
    return np.asarray(bunch.data)


def _load_uci_dataset(args: argparse.Namespace) -> np.ndarray:
    if args.uci_dataset_id is None:
        raise ValueError("UCI sources require --uci-dataset-id.")

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "UCI dataset sources require the 'ucimlrepo' package."
        ) from exc

    dataset = fetch_ucirepo(id=args.uci_dataset_id)
    features = dataset.data.features
    if args.feature_columns:
        return _coerce_columnar_matrix(features, list(args.feature_columns))
    return np.asarray(features)


def _load_huggingface_dataset(args: argparse.Namespace) -> np.ndarray:
    if not args.hf_repo:
        raise ValueError("Hugging Face sources require --hf-repo.")
    if not args.hf_feature_column and not args.feature_columns:
        raise ValueError(
            "Hugging Face sources require --hf-feature-column or --feature-columns."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset sources require the 'datasets' package."
        ) from exc

    load_args = [args.hf_repo]
    if args.hf_config:
        load_args.append(args.hf_config)

    dataset = load_dataset(
        *load_args,
        split=args.hf_split,
        cache_dir=str(Path(args.downloads_dir).expanduser() / "huggingface"),
    )

    if args.hf_feature_column:
        return _coerce_vector_column(dataset, args.hf_feature_column)

    return _coerce_columnar_matrix(dataset, list(args.feature_columns))


def _validate_dataset_array(
    X: np.ndarray,
    *,
    dataset: str,
    N: int,
    D: int,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(
            f"Dataset {dataset!r} must be a dense 2-D array, got shape {X.shape!r}."
        )
    if X.shape != (N, D):
        raise ValueError(
            f"Dataset {dataset!r} has shape {X.shape!r}, expected ({N}, {D})."
        )
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(
            f"Dataset {dataset!r} must be numeric, got dtype {X.dtype!r}."
        )

    X_float32 = np.ascontiguousarray(X, dtype=np.float32)
    if not np.all(np.isfinite(X_float32)):
        raise ValueError(f"Dataset {dataset!r} contains NaN or infinite values.")

    return X_float32


def load_dataset(args: argparse.Namespace) -> np.ndarray:
    validate_dataset_key(args.dataset)

    if args.K <= 0:
        raise ValueError(f"K must be positive, got {args.K}.")
    if args.K > args.N:
        raise ValueError(f"K must not exceed N, got K={args.K}, N={args.N}.")

    if args.source_kind == "blobs":
        raw = _load_blob_dataset(args.N, args.D, args.K)
    elif args.source_kind in {"local"}:
        if not args.source_path:
            raise ValueError(
                f"--source-path is required for source kind {args.source_kind!r}."
            )
        source_format = _source_format_for_args(args)
        if source_format is None:
            raise ValueError(
                "Local dataset sources require --source-format."
            )
        raw = _load_local_dataset(args.source_path, source_format, args.N, args.D)
    elif args.source_kind == "url":
        if not args.source_url:
            raise ValueError("--source-url is required for URL dataset sources.")
        source_format = _source_format_for_args(args)
        if source_format is None:
            raise ValueError(
                "URL dataset sources require --source-format unless the URL "
                "ends in .npy or .npz."
            )
        downloaded_path = _download_url(
            args.source_url,
            args.downloads_dir,
            args.dataset,
        )
        raw = _load_local_dataset(downloaded_path, source_format, args.N, args.D)
    elif args.source_kind == "openml":
        raw = _load_openml_dataset(args)
    elif args.source_kind == "uci":
        raw = _load_uci_dataset(args)
    elif args.source_kind == "huggingface":
        raw = _load_huggingface_dataset(args)
    else:
        raise ValueError(f"Unsupported source kind: {args.source_kind!r}")

    return _validate_dataset_array(raw, dataset=args.dataset, N=args.N, D=args.D)


def main() -> None:
    args = parse_args()

    ensure_parent_dir(args.dataset_out)
    ensure_parent_dir(args.centroids_out)

    # 1. Materialize the benchmark dataset in the shared row-major float32 format.
    X_float32 = load_dataset(args)
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

    ensure_parent_dir(args.gmm_weights_out)
    ensure_parent_dir(args.gmm_means_out)
    for path in precision_outputs.values():
        ensure_parent_dir(path)

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
