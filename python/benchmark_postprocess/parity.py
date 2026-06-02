from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark_postprocess.io import load_json
from benchmark_postprocess.naming import (
    GMM_METRICS_JSON_RE,
    LLOYD_PARITY_JSON_RE,
    format_config_id,
    parse_config_match,
)

REQUIRED_LANGUAGE_KEYS = ("cpp", "py")

GMM_PARITY_THRESHOLDS = {
    "lower_bound_diff_abs": 1e-4,
    "weights_max_abs_diff": 1e-4,
    "means_max_abs_diff": 1e-3,
    "covariances_max_rel_diff": 1e-2,
    "algorithm_iteration_diff_abs": 1,
}


def lloyd_algorithm_iteration_count(
    lloyd_parity: dict[str, dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
) -> int:
    parity = lloyd_parity.get(config_id)

    if parity is None:
        raise RuntimeError(
            f"Missing Lloyd parity file for {config_id}. "
            f"The orchestrator should finalize each config before post-processing."
        )

    if lang_key == "cpp":
        return int(parity["cpp_algorithm_iterations"])

    if lang_key == "py":
        return int(parity["python_algorithm_iterations"])

    raise RuntimeError(f"Unexpected Lloyd language key: {lang_key!r}")


def gmm_algorithm_iteration_count(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
    lang_key: str,
) -> int:
    metrics = gmm_metrics.get((config_id, lang_key))

    if metrics is None:
        raise RuntimeError(
            f"Missing GMM metrics file for {lang_key} {config_id}. "
            "GMM timing needs the EM algorithm-iteration count to report time_per_algorithm_iteration_s."
        )

    return int(metrics["algorithm_iterations"])


def normalize_lloyd_parity_record(
    record: dict[str, Any],
    *,
    config_id: str,
) -> dict[str, Any]:
    required_fields = [
        "cpp_algorithm_iterations",
        "python_algorithm_iterations",
        "cpp_inertia",
        "python_inertia",
        "inertia_diff_abs",
        "inertia_diff_pct",
        "tolerance_pct",
        "status",
    ]

    for field in required_fields:
        if field not in record:
            raise RuntimeError(f"Missing {field!r} in Lloyd parity for {config_id}")

    record_config_id = record.get("config_id")

    if record_config_id is not None and record_config_id != config_id:
        raise RuntimeError(
            f"Lloyd parity config mismatch: file is for {record_config_id}, "
            f"expected {config_id}"
        )

    return {
        **record,
        "config_id": config_id,
        "cpp_algorithm_iterations": int(record["cpp_algorithm_iterations"]),
        "python_algorithm_iterations": int(record["python_algorithm_iterations"]),
        "cpp_inertia": float(record["cpp_inertia"]),
        "python_inertia": float(record["python_inertia"]),
        "inertia_diff_abs": float(record["inertia_diff_abs"]),
        "inertia_diff_pct": float(record["inertia_diff_pct"]),
        "tolerance_pct": float(record["tolerance_pct"]),
        "status": str(record["status"]),
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
        "converged",
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
        "converged": bool(record["converged"]),
        "lower_bound": float(record["lower_bound"]),
        "covariance_type": str(record["covariance_type"]),
    }


def load_lloyd_parity_map(data_dir: Path) -> dict[str, dict[str, Any]]:
    parity_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("lloyd_parity_*.json")):
        match = LLOYD_PARITY_JSON_RE.match(path.name)

        if not match:
            print(f"Skipping malformed Lloyd parity filename: {path.name}")
            continue

        D, N, K = parse_config_match(match)
        config_id = format_config_id(D, N, K)

        parity = normalize_lloyd_parity_record(
            load_json(path),
            config_id=config_id,
        )

        if int(parity.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported Lloyd parity schema in {path}")

        parity_records[config_id] = parity

        if parity["status"] != "PASS":
            failures.append(parity)

    if failures:
        failed_ids = ", ".join(record["config_id"] for record in failures[:10])

        print(
            f"WARNING: Loaded {len(failures)} Lloyd parity failures. "
            f"First failures: {failed_ids}"
        )

    pass_count = sum(
        1 for record in parity_records.values() if record["status"] == "PASS"
    )
    fail_count = sum(
        1 for record in parity_records.values() if record["status"] != "PASS"
    )

    print(
        f"Loaded {len(parity_records)} Lloyd parity records "
        f"({pass_count} PASS, {fail_count} non-PASS)."
    )

    return parity_records


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


def gmm_completed_config_ids(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    """Return configs with enough GMM metrics for timing and C++/Python speedups."""
    by_config: dict[str, set[str]] = {}

    for config_id, lang_key in gmm_metrics:
        by_config.setdefault(config_id, set()).add(lang_key)

    required = set(required_languages)
    return {
        config_id
        for config_id, languages in by_config.items()
        if required.issubset(languages)
    }


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


def compute_gmm_comparison(
    gmm_metrics: dict[tuple[str, str], dict[str, Any]],
    *,
    config_id: str,
) -> dict[str, Any]:
    """Build GMM parity info using fixed project-level tolerances.

    Unlike Lloyd, GMM has no inertia. The main numerical target is the final
    average lower bound, plus sanity checks on convergence, algorithm-iteration count,
    and learned parameters.
    """
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

    converged_match = bool(cpp["converged"]) == bool(py["converged"])

    checks = {
        "converged_match": converged_match,
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

    failure_reasons = [
        check_name for check_name, passed in checks.items() if not passed
    ]

    status = "PASS" if not failure_reasons else "FAIL"

    return {
        "status": status,
        "failure_reasons": failure_reasons,
        "checks": checks,
        "thresholds": GMM_PARITY_THRESHOLDS,
        "cpp_algorithm_iterations": cpp_algorithm_iterations,
        "python_algorithm_iterations": py_algorithm_iterations,
        "algorithm_iteration_diff_abs": algorithm_iteration_diff_abs,
        "cpp_converged": bool(cpp["converged"]),
        "python_converged": bool(py["converged"]),
        "converged_match": converged_match,
        "covariance_type": str(cpp["covariance_type"]),
        "cpp_lower_bound": cpp_lower_bound,
        "python_lower_bound": py_lower_bound,
        "lower_bound_diff_abs": lower_bound_diff_abs,
        "lower_bound_diff_pct": lower_bound_diff_pct,
        "weights_max_abs_diff": weights_max_abs_diff,
        "means_max_abs_diff": means_max_abs_diff,
        "covariances_max_rel_diff": covariances_max_rel_diff,
    }
