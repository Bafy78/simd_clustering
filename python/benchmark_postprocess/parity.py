from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark_postprocess.io import load_json
from benchmark_postprocess.naming import (
    GMM_METRICS_JSON_RE,
    LLOYD_METRICS_JSON_RE,
    format_config_id,
    parse_config_match,
)

REQUIRED_LANGUAGE_KEYS = ("cpp", "py")

LLOYD_PARITY_THRESHOLDS = {
    "inertia_diff_pct": 1e-6,
    "algorithm_iteration_diff_abs": 0,
}

GMM_PARITY_THRESHOLDS = {
    "lower_bound_diff_abs": 1e-4,
    "weights_max_abs_diff": 1e-4,
    "means_max_abs_diff": 1e-3,
    "covariances_max_rel_diff": 1e-2,
    "algorithm_iteration_diff_abs": 0,
}


def _language_metrics(
    metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
    phase_name: str,
) -> dict[str, Any]:
    record = metrics.get((config_id, lang_key))

    if record is None:
        raise RuntimeError(
            f"Missing {phase_name} metrics file for {lang_key} {config_id}. "
            f"{phase_name} timing needs the algorithm-iteration count to report "
            "time_per_algorithm_iteration_s."
        )

    return record


def lloyd_algorithm_iteration_count(
    lloyd_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
) -> int:
    return int(
        _language_metrics(
            lloyd_metrics,
            config_id=config_id,
            lang_key=lang_key,
            phase_name="Lloyd",
        )["algorithm_iterations"]
    )


def gmm_algorithm_iteration_count(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
) -> int:
    return int(
        _language_metrics(
            gmm_metrics,
            config_id=config_id,
            lang_key=lang_key,
            phase_name="GMM",
        )["algorithm_iterations"]
    )


def normalize_lloyd_metrics_record(
    record: dict[str, Any],
    *,
    config_id: str,
    lang_key: str,
) -> dict[str, Any]:
    required_fields = [
        "phase",
        "language",
        "algorithm_iterations",
        "inertia",
        "cluster_counts",
        "cluster_inertia",
    ]

    for field in required_fields:
        if field not in record:
            raise RuntimeError(f"Missing {field!r} in Lloyd metrics for {config_id}")

    if record["phase"] != "lloyd":
        raise RuntimeError(f"Unexpected phase in Lloyd metrics for {config_id}")

    if record["language"] != lang_key:
        raise RuntimeError(
            f"Lloyd metrics language mismatch: file is {lang_key}, "
            f"record says {record['language']!r}"
        )

    return {
        **record,
        "config_id": config_id,
        "language": lang_key,
        "algorithm_iterations": int(record["algorithm_iterations"]),
        "inertia": float(record["inertia"]),
    }


def normalize_gmm_metrics_record(
    record: dict[str, Any],
    *,
    config_id: str,
    lang_key: str,
) -> dict[str, Any]:
    required_fields = [
        "phase",
        "language",
        "covariance_type",
        "algorithm_iterations",
        "lower_bound",
    ]

    for field in required_fields:
        if field not in record:
            raise RuntimeError(f"Missing {field!r} in GMM metrics for {config_id}")

    if record["phase"] != "gmm":
        raise RuntimeError(f"Unexpected phase in GMM metrics for {config_id}")

    if record["language"] != lang_key:
        raise RuntimeError(
            f"GMM metrics language mismatch: file is {lang_key}, "
            f"record says {record['language']!r}"
        )

    return {
        **record,
        "config_id": config_id,
        "language": lang_key,
        "algorithm_iterations": int(record["algorithm_iterations"]),
        "lower_bound": float(record["lower_bound"]),
        "covariance_type": str(record["covariance_type"]),
    }


def load_lloyd_metrics_map(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    metrics_records: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(data_dir.glob("lloyd_metrics_*.json")):
        match = LLOYD_METRICS_JSON_RE.match(path.name)

        if not match:
            print(f"Skipping malformed Lloyd metrics filename: {path.name}")
            continue

        lang_key = match.group("lang")
        D, N, K = parse_config_match(match)
        config_id = format_config_id(D, N, K)

        metrics = normalize_lloyd_metrics_record(
            load_json(path),
            config_id=config_id,
            lang_key=lang_key,
        )

        if int(metrics.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported Lloyd metrics schema in {path}")

        metrics_records[(config_id, lang_key)] = metrics

    print(f"Loaded {len(metrics_records)} Lloyd metrics records.")

    return metrics_records


def load_gmm_metrics_map(data_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    metrics_records: dict[tuple[str, str], dict[str, Any]] = {}

    for path in sorted(data_dir.glob("gmm_metrics_*.json")):
        match = GMM_METRICS_JSON_RE.match(path.name)

        if not match:
            print(f"Skipping malformed GMM metrics filename: {path.name}")
            continue

        lang_key = match.group("lang")
        D, N, K = parse_config_match(match)
        config_id = format_config_id(D, N, K)

        metrics = normalize_gmm_metrics_record(
            load_json(path),
            config_id=config_id,
            lang_key=lang_key,
        )

        if int(metrics.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported GMM metrics schema in {path}")

        metrics_records[(config_id, lang_key)] = metrics

    print(f"Loaded {len(metrics_records)} GMM metrics records.")

    return metrics_records


def completed_config_ids(
    metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    """Return configs with enough metrics for timing and C++/Python speedups."""
    by_config: dict[str, set[str]] = {}

    for config_id, lang_key in metrics:
        by_config.setdefault(config_id, set()).add(lang_key)

    required = set(required_languages)
    return {
        config_id
        for config_id, languages in by_config.items()
        if required.issubset(languages)
    }


def lloyd_completed_config_ids(
    lloyd_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    return completed_config_ids(lloyd_metrics, required_languages=required_languages)


def gmm_completed_config_ids(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    return completed_config_ids(gmm_metrics, required_languages=required_languages)


def _relative_diff_pct(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale * 100.0


def _max_abs_diff(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference is None:
        return None

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    if cand.shape != ref.shape:
        return None

    if cand.size == 0:
        return 0.0

    return float(np.max(np.abs(cand - ref)))


def _max_rel_diff(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference is None:
        return None

    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)

    if cand.shape != ref.shape:
        return None

    if cand.size == 0:
        return 0.0

    scale = np.maximum(np.abs(ref), np.finfo(np.float64).eps)
    return float(np.max(np.abs(cand - ref) / scale))


def _is_finite_and_within(value: float | None, tolerance: float) -> bool:
    if value is None:
        return False

    return bool(np.isfinite(value) and value <= tolerance)


def _status_from_checks(checks: dict[str, bool]) -> tuple[str, list[str]]:
    failure_reasons = [
        check_name for check_name, passed in checks.items() if not passed
    ]
    return ("PASS" if not failure_reasons else "FAIL", failure_reasons)


def compute_lloyd_comparison(
    lloyd_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
) -> dict[str, Any]:
    cpp = lloyd_metrics.get((config_id, "cpp"))
    py = lloyd_metrics.get((config_id, "py"))

    if cpp is None or py is None:
        missing = [
            lang_key
            for lang_key, record in (("cpp", cpp), ("py", py))
            if record is None
        ]
        raise RuntimeError(f"Missing Lloyd metrics for {config_id}: {', '.join(missing)}")

    cpp_algorithm_iterations = int(cpp["algorithm_iterations"])
    py_algorithm_iterations = int(py["algorithm_iterations"])
    algorithm_iteration_diff_abs = abs(
        cpp_algorithm_iterations - py_algorithm_iterations
    )

    cpp_inertia = float(cpp["inertia"])
    py_inertia = float(py["inertia"])
    inertia_diff_abs = abs(cpp_inertia - py_inertia)
    scale = max(abs(cpp_inertia), abs(py_inertia))
    inertia_diff_pct = inertia_diff_abs / scale * 100.0 if scale > 0.0 else 0.0

    checks = {
        "inertia_diff_pct": _is_finite_and_within(
            inertia_diff_pct,
            LLOYD_PARITY_THRESHOLDS["inertia_diff_pct"],
        ),
        "algorithm_iteration_diff_abs": (
            algorithm_iteration_diff_abs
            <= LLOYD_PARITY_THRESHOLDS["algorithm_iteration_diff_abs"]
        ),
    }
    status, failure_reasons = _status_from_checks(checks)

    return {
        "config_id": config_id,
        "status": status,
        "failure_reasons": failure_reasons,
        "checks": checks,
        "thresholds": dict(LLOYD_PARITY_THRESHOLDS),
        "cpp_algorithm_iterations": cpp_algorithm_iterations,
        "python_algorithm_iterations": py_algorithm_iterations,
        "algorithm_iteration_diff_abs": algorithm_iteration_diff_abs,
        "cpp_inertia": cpp_inertia,
        "python_inertia": py_inertia,
        "inertia_diff_abs": inertia_diff_abs,
        "inertia_diff_pct": inertia_diff_pct,
        "cpp_cluster_counts": cpp.get("cluster_counts"),
        "python_cluster_counts": py.get("cluster_counts"),
        "cpp_cluster_inertia": cpp.get("cluster_inertia"),
        "python_cluster_inertia": py.get("cluster_inertia"),
    }


def compute_gmm_comparison(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
) -> dict[str, Any]:
    """Build GMM parity info from raw C++ and sklearn metrics."""
    cpp = gmm_metrics.get((config_id, "cpp"))
    py = gmm_metrics.get((config_id, "py"))

    if cpp is None or py is None:
        missing = [
            lang_key
            for lang_key, record in (("cpp", cpp), ("py", py))
            if record is None
        ]
        raise RuntimeError(f"Missing GMM metrics for {config_id}: {', '.join(missing)}")

    cpp_algorithm_iterations = int(cpp["algorithm_iterations"])
    py_algorithm_iterations = int(py["algorithm_iterations"])
    algorithm_iteration_diff_abs = abs(
        cpp_algorithm_iterations - py_algorithm_iterations
    )

    cpp_lower_bound = float(cpp["lower_bound"])
    py_lower_bound = float(py["lower_bound"])

    lower_bound_diff_abs = abs(cpp_lower_bound - py_lower_bound)
    lower_bound_diff_pct = _relative_diff_pct(cpp_lower_bound, py_lower_bound)

    weights_max_abs_diff = _max_abs_diff(cpp.get("weights"), py.get("weights"))
    means_max_abs_diff = _max_abs_diff(cpp.get("means"), py.get("means"))
    covariances_max_rel_diff = _max_rel_diff(
        cpp.get("covariances"),
        py.get("covariances"),
    )

    checks = {
        "algorithm_iteration_diff_abs": (
            algorithm_iteration_diff_abs
            <= GMM_PARITY_THRESHOLDS["algorithm_iteration_diff_abs"]
        ),
        "lower_bound_diff_abs": _is_finite_and_within(
            lower_bound_diff_abs,
            GMM_PARITY_THRESHOLDS["lower_bound_diff_abs"],
        ),
        "weights_max_abs_diff": _is_finite_and_within(
            weights_max_abs_diff,
            GMM_PARITY_THRESHOLDS["weights_max_abs_diff"],
        ),
        "means_max_abs_diff": _is_finite_and_within(
            means_max_abs_diff,
            GMM_PARITY_THRESHOLDS["means_max_abs_diff"],
        ),
        "covariances_max_rel_diff": _is_finite_and_within(
            covariances_max_rel_diff,
            GMM_PARITY_THRESHOLDS["covariances_max_rel_diff"],
        ),
    }
    status, failure_reasons = _status_from_checks(checks)

    return {
        "config_id": config_id,
        "status": status,
        "failure_reasons": failure_reasons,
        "checks": checks,
        "thresholds": dict(GMM_PARITY_THRESHOLDS),
        "cpp_algorithm_iterations": cpp_algorithm_iterations,
        "python_algorithm_iterations": py_algorithm_iterations,
        "algorithm_iteration_diff_abs": algorithm_iteration_diff_abs,
        "covariance_type": str(cpp["covariance_type"]),
        "cpp_lower_bound": cpp_lower_bound,
        "python_lower_bound": py_lower_bound,
        "lower_bound_diff_abs": lower_bound_diff_abs,
        "lower_bound_diff_pct": lower_bound_diff_pct,
        "weights_max_abs_diff": weights_max_abs_diff,
        "means_max_abs_diff": means_max_abs_diff,
        "covariances_max_rel_diff": covariances_max_rel_diff,
    }
