from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmark_postprocess.io import load_json
from benchmark_metadata import (
    FULL_STAGE_KEY,
    LANGUAGE_CPP_KEY,
    LANGUAGE_PY_KEY,
    NO_PARAMS,
    REFERENCE_VARIANT,
)
from benchmark_postprocess.naming import (
    BenchmarkIdentity,
    MetricsKey,
    parse_metrics_filename,
    variant_display_name,
)

REQUIRED_LANGUAGE_KEYS = (LANGUAGE_CPP_KEY, LANGUAGE_PY_KEY)

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

HDBSCAN_STAGE_PARITY_THRESHOLDS = {
    "diagonal_max_abs": 1e-12,
    "symmetry_max_abs": 1e-12,
    "summary_scalar_abs_diff": 1e-3,
    "summary_scalar_rel_diff": 1e-5,
    "probe_value_max_abs_diff": 1e-4,
}


def metrics_key(
    config_id: str,
    stage_key: str,
    variant_key: str,
    lang_key: str,
    params_key: str = NO_PARAMS,
) -> MetricsKey:
    return (config_id, stage_key, variant_key, lang_key, params_key)


def _language_metrics(
    metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    stage_key: str,
    variant_key: str,
    lang_key: str,
    phase_name: str,
    params_key: str = NO_PARAMS,
) -> dict[str, Any]:
    record = metrics.get(
        metrics_key(config_id, stage_key, variant_key, lang_key, params_key)
    )

    if record is not None:
        return record

    raise RuntimeError(
        f"Missing {phase_name} metrics file for {lang_key} "
        f"stage={stage_key} variant={variant_key} params={params_key} {config_id}. {phase_name} timing needs the "
        "algorithm-iteration count to report time_per_algorithm_iteration_s."
    )


def lloyd_algorithm_iteration_count(
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    variant_key: str,
    lang_key: str,
    params_key: str = NO_PARAMS,
    stage_key: str = FULL_STAGE_KEY,
) -> int:
    return int(
        _language_metrics(
            lloyd_metrics,
            config_id=config_id,
            stage_key=stage_key,
            variant_key=variant_key,
            lang_key=lang_key,
            phase_name="Lloyd",
            params_key=params_key,
        )["algorithm_iterations"]
    )


def gmm_algorithm_iteration_count(
    gmm_metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    variant_key: str,
    lang_key: str,
    params_key: str,
    stage_key: str = FULL_STAGE_KEY,
) -> int:
    return int(
        _language_metrics(
            gmm_metrics,
            config_id=config_id,
            stage_key=stage_key,
            variant_key=variant_key,
            lang_key=lang_key,
            phase_name="GMM",
            params_key=params_key,
        )["algorithm_iterations"]
    )


def normalize_lloyd_metrics_record(
    record: dict[str, Any],
    identity: BenchmarkIdentity,
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
            raise RuntimeError(
                f"Missing {field!r} in Lloyd metrics for {identity.config_id}"
            )

    if record["phase"] != "lloyd":
        raise RuntimeError(
            f"Unexpected phase in Lloyd metrics for {identity.config_id}"
        )

    if record["language"] != identity.language_key:
        raise RuntimeError(
            f"Lloyd metrics language mismatch: file is {identity.language_key}, "
            f"record says {record['language']!r}"
        )

    return {
        **record,
        "config_id": identity.config_id,
        "stage_key": identity.stage_key,
        "stage": identity.stage,
        "variant_key": identity.variant_key,
        "variant": identity.variant,
        "params_key": identity.params_key,
        "language": identity.language_key,
        "algorithm_iterations": int(record["algorithm_iterations"]),
        "inertia": float(record["inertia"]),
    }


def normalize_gmm_metrics_record(
    record: dict[str, Any],
    identity: BenchmarkIdentity,
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
            raise RuntimeError(
                f"Missing {field!r} in GMM metrics for {identity.config_id}"
            )

    if record["phase"] != "gmm":
        raise RuntimeError(
            f"Unexpected phase in GMM metrics for {identity.config_id}"
        )

    if record["language"] != identity.language_key:
        raise RuntimeError(
            f"GMM metrics language mismatch: file is {identity.language_key}, "
            f"record says {record['language']!r}"
        )

    if str(record["covariance_type"]) != identity.params_key:
        raise RuntimeError(
            f"GMM metrics covariance mismatch: file params={identity.params_key!r}, "
            f"record says {record['covariance_type']!r}"
        )

    return {
        **record,
        "config_id": identity.config_id,
        "stage_key": identity.stage_key,
        "stage": identity.stage,
        "variant_key": identity.variant_key,
        "variant": identity.variant,
        "params_key": identity.params_key,
        "language": identity.language_key,
        "algorithm_iterations": int(record["algorithm_iterations"]),
        "lower_bound": float(record["lower_bound"]),
        "covariance_type": str(record["covariance_type"]),
    }


def normalize_hdbscan_metrics_record(
    record: dict[str, Any],
    identity: BenchmarkIdentity,
) -> dict[str, Any]:
    required_fields = [
        "phase",
        "language",
        "stage",
        "dtype",
        "shape",
        "summary",
    ]

    for field in required_fields:
        if field not in record:
            raise RuntimeError(
                f"Missing {field!r} in HDBSCAN metrics for {identity.config_id}"
            )

    if record["phase"] != "hdbscan":
        raise RuntimeError(
            f"Unexpected phase in HDBSCAN metrics for {identity.config_id}"
        )

    if record["language"] != identity.language_key:
        raise RuntimeError(
            f"HDBSCAN metrics language mismatch: file is {identity.language_key}, "
            f"record says {record['language']!r}"
        )

    if record["stage"] != identity.stage_key:
        raise RuntimeError(
            f"HDBSCAN metrics stage mismatch: file is {identity.stage_key}, "
            f"record says {record['stage']!r}"
        )

    return {
        **record,
        "config_id": identity.config_id,
        "stage_key": identity.stage_key,
        "variant_key": identity.variant_key,
        "variant": identity.variant,
        "params_key": identity.params_key,
        "language": identity.language_key,
    }


def load_lloyd_metrics_map(data_dir: Path) -> dict[MetricsKey, dict[str, Any]]:
    metrics_records: dict[MetricsKey, dict[str, Any]] = {}

    for path in sorted(set(data_dir.glob("lloyd_metrics_*.json")) | set(data_dir.glob("lloyd_*_metrics_*.json"))):
        identity = parse_metrics_filename(path, "lloyd")

        if identity is None:
            print(f"Skipping malformed Lloyd metrics filename: {path.name}")
            continue

        metrics = normalize_lloyd_metrics_record(load_json(path), identity)

        if int(metrics.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported Lloyd metrics schema in {path}")

        metrics_records[identity.metrics_key] = metrics

    print(f"Loaded {len(metrics_records)} Lloyd metrics records.")

    return metrics_records


def load_gmm_metrics_map(data_dir: Path) -> dict[MetricsKey, dict[str, Any]]:
    metrics_records: dict[MetricsKey, dict[str, Any]] = {}

    for path in sorted(set(data_dir.glob("gmm_metrics_*.json")) | set(data_dir.glob("gmm_*_metrics_*.json"))):
        identity = parse_metrics_filename(path, "gmm")

        if identity is None:
            print(f"Skipping malformed GMM metrics filename: {path.name}")
            continue

        metrics = normalize_gmm_metrics_record(load_json(path), identity)

        if int(metrics.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported GMM metrics schema in {path}")

        metrics_records[identity.metrics_key] = metrics

    print(f"Loaded {len(metrics_records)} GMM metrics records.")

    return metrics_records


def load_hdbscan_metrics_map(data_dir: Path) -> dict[MetricsKey, dict[str, Any]]:
    metrics_records: dict[MetricsKey, dict[str, Any]] = {}

    for path in sorted(data_dir.glob("hdbscan_*_metrics_*.json")):
        identity = parse_metrics_filename(path, "hdbscan")

        if identity is None:
            print(f"Skipping malformed HDBSCAN metrics filename: {path.name}")
            continue

        metrics = normalize_hdbscan_metrics_record(load_json(path), identity)

        if int(metrics.get("schema_version", 1)) != 1:
            raise RuntimeError(f"Unsupported HDBSCAN metrics schema in {path}")

        metrics_records[identity.metrics_key] = metrics

    print(f"Loaded {len(metrics_records)} HDBSCAN metrics records.")

    return metrics_records


def hdbscan_completed_config_ids(
    hdbscan_metrics: dict[MetricsKey, dict[str, Any]],
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    return completed_config_ids(hdbscan_metrics, required_languages=required_languages)


def completed_metric_keys(
    metrics: dict[MetricsKey, dict[str, Any]],
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[tuple[str, str, str]]:
    """Return (config_id, stage_key, params_key) keys with C++ plus reference metrics."""
    by_config_params: dict[tuple[str, str, str], set[str]] = {}

    for config_id, stage_key, _variant_key, lang_key, params_key in metrics:
        by_config_params.setdefault((config_id, stage_key, params_key), set()).add(lang_key)

    required = set(required_languages)
    return {
        key
        for key, languages in by_config_params.items()
        if required.issubset(languages)
    }


def completed_config_ids(
    metrics: dict[MetricsKey, dict[str, Any]],
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    return {
        config_id
        for config_id, _stage_key, _params_key in completed_metric_keys(
            metrics,
            required_languages=required_languages,
        )
    }


def lloyd_completed_config_ids(
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    required_languages: tuple[str, ...] = REQUIRED_LANGUAGE_KEYS,
) -> set[str]:
    return completed_config_ids(lloyd_metrics, required_languages=required_languages)


def gmm_completed_config_ids(
    gmm_metrics: dict[MetricsKey, dict[str, Any]],
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
    lloyd_metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    cpp_variant_key: str,
    py_variant_key: str = REFERENCE_VARIANT,
    params_key: str = NO_PARAMS,
    stage_key: str = FULL_STAGE_KEY,
) -> dict[str, Any]:
    cpp = _language_metrics(
        lloyd_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=cpp_variant_key,
        lang_key=LANGUAGE_CPP_KEY,
        phase_name="Lloyd",
        params_key=params_key,
    )
    py = _language_metrics(
        lloyd_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=py_variant_key,
        lang_key=LANGUAGE_PY_KEY,
        phase_name="Lloyd",
        params_key=params_key,
    )

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
        "stage_key": stage_key,
        "stage": cpp.get("stage"),
        "variant_key": cpp_variant_key,
        "variant": variant_display_name(cpp_variant_key),
        "params_key": params_key,
        "python_variant_key": py.get("variant_key", py_variant_key),
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
    gmm_metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    cpp_variant_key: str,
    params_key: str,
    py_variant_key: str = REFERENCE_VARIANT,
    stage_key: str = FULL_STAGE_KEY,
) -> dict[str, Any]:
    """Build GMM parity info from one C++ variant and the shared sklearn reference."""
    cpp = _language_metrics(
        gmm_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=cpp_variant_key,
        lang_key=LANGUAGE_CPP_KEY,
        phase_name="GMM",
        params_key=params_key,
    )
    py = _language_metrics(
        gmm_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=py_variant_key,
        lang_key=LANGUAGE_PY_KEY,
        phase_name="GMM",
        params_key=params_key,
    )

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
        "stage_key": stage_key,
        "stage": cpp.get("stage"),
        "variant_key": cpp_variant_key,
        "variant": variant_display_name(cpp_variant_key),
        "params_key": params_key,
        "python_variant_key": py.get("variant_key", py_variant_key),
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



def _summary_scalar_diff(candidate_summary: dict[str, Any], reference_summary: dict[str, Any], field: str) -> dict[str, float]:
    candidate = float(candidate_summary[field])
    reference = float(reference_summary[field])
    abs_diff = abs(candidate - reference)
    scale = max(abs(candidate), abs(reference), 1.0)
    return {
        "candidate": candidate,
        "reference": reference,
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / scale,
    }


def _probe_value_max_abs_diff(candidate_summary: dict[str, Any], reference_summary: dict[str, Any]) -> float | None:
    candidate_probes = candidate_summary.get("probes", [])
    reference_probes = reference_summary.get("probes", [])

    if len(candidate_probes) != len(reference_probes):
        return None

    max_abs_diff = 0.0
    for candidate_probe, reference_probe in zip(candidate_probes, reference_probes):
        if int(candidate_probe.get("index", -1)) != int(reference_probe.get("index", -2)):
            return None
        max_abs_diff = max(
            max_abs_diff,
            abs(float(candidate_probe["value"]) - float(reference_probe["value"])),
        )
    return max_abs_diff


def _summary_comparison(
    candidate_summary: dict[str, Any],
    reference_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scalar_diffs = {
        field: _summary_scalar_diff(candidate_summary, reference_summary, field)
        for field in ("sum", "sum_abs", "sum_squares", "weighted_sum", "min", "max")
    }
    scalar_abs_diff_max = max(diff["abs_diff"] for diff in scalar_diffs.values())
    scalar_rel_diff_max = max(diff["rel_diff"] for diff in scalar_diffs.values())
    probe_value_max_abs_diff = _probe_value_max_abs_diff(candidate_summary, reference_summary)

    summary_count_fields = (
        "value_count",
        "finite_count",
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
    )
    summary_counts_equal = all(
        candidate_summary.get(field) == reference_summary.get(field)
        for field in summary_count_fields
    )

    checks = {
        "summary_counts": summary_counts_equal,
        "summary_scalar_abs_or_rel_diff": (
            scalar_abs_diff_max
            <= HDBSCAN_STAGE_PARITY_THRESHOLDS["summary_scalar_abs_diff"]
            or scalar_rel_diff_max
            <= HDBSCAN_STAGE_PARITY_THRESHOLDS["summary_scalar_rel_diff"]
        ),
        "probe_value_max_abs_diff": _is_finite_and_within(
            probe_value_max_abs_diff,
            HDBSCAN_STAGE_PARITY_THRESHOLDS["probe_value_max_abs_diff"],
        ),
    }
    details = {
        "scalar_diffs": scalar_diffs,
        "scalar_abs_diff_max": scalar_abs_diff_max,
        "scalar_rel_diff_max": scalar_rel_diff_max,
        "probe_value_max_abs_diff": probe_value_max_abs_diff,
        "candidate_hash": candidate_summary.get("fnv1a64_float64"),
        "reference_hash": reference_summary.get("fnv1a64_float64"),
        "hash_equal": (
            candidate_summary.get("fnv1a64_float64")
            == reference_summary.get("fnv1a64_float64")
        ),
    }
    return checks, details


def compute_hdbscan_comparison(
    hdbscan_metrics: dict[MetricsKey, dict[str, Any]],
    config_id: str,
    cpp_variant_key: str,
    py_variant_key: str = REFERENCE_VARIANT,
    params_key: str = NO_PARAMS,
    stage_key: str = FULL_STAGE_KEY,
) -> dict[str, Any]:
    cpp = _language_metrics(
        hdbscan_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=cpp_variant_key,
        lang_key=LANGUAGE_CPP_KEY,
        phase_name="HDBSCAN",
        params_key=params_key,
    )
    py = _language_metrics(
        hdbscan_metrics,
        config_id=config_id,
        stage_key=stage_key,
        variant_key=py_variant_key,
        lang_key=LANGUAGE_PY_KEY,
        phase_name="HDBSCAN",
        params_key=params_key,
    )

    cpp_summary = cpp.get("summary", {})
    py_summary = py.get("summary", {})
    summary_checks, summary_details = _summary_comparison(cpp_summary, py_summary)

    checks = dict(summary_checks)
    diagonal_details = None
    core_distance_details = None
    label_details = None
    probability_details = None

    if stage_key == "distance":
        core_distance_checks, core_distance_details = _summary_comparison(
            cpp.get("core_distance_summary", {}),
            py.get("core_distance_summary", {}),
        )
        checks.update(
            {f"core_distance_{name}": value for name, value in core_distance_checks.items()}
        )
    elif stage_key in {"select", "full"}:
        label_checks, label_details = _summary_comparison(
            cpp.get("label_summary", {}),
            py.get("label_summary", {}),
        )
        probability_checks, probability_details = _summary_comparison(
            cpp.get("probability_summary", {}),
            py.get("probability_summary", {}),
        )
        checks.update(
            {
                "noise_count_equal": int(cpp.get("noise_count", -1)) == int(py.get("noise_count", -2)),
                "cluster_count_equal": int(cpp.get("cluster_count", -1)) == int(py.get("cluster_count", -2)),
                **{f"label_{name}": value for name, value in label_checks.items()},
                **{f"probability_{name}": value for name, value in probability_checks.items()},
            }
        )

    status, failure_reasons = _status_from_checks(checks)

    result = {
        "config_id": config_id,
        "stage_key": stage_key,
        "stage": cpp.get("stage", stage_key),
        "variant_key": cpp_variant_key,
        "variant": variant_display_name(cpp_variant_key),
        "params_key": params_key,
        "python_variant_key": py.get("variant_key", py_variant_key),
        "status": status,
        "failure_reasons": failure_reasons,
        "checks": checks,
        "thresholds": dict(HDBSCAN_STAGE_PARITY_THRESHOLDS),
        "cpp_shape": cpp.get("shape"),
        "python_shape": py.get("shape"),
        "cpp_hash": summary_details["candidate_hash"],
        "python_hash": summary_details["reference_hash"],
        "hash_equal": summary_details["hash_equal"],
        "summary_scalar_diffs": summary_details["scalar_diffs"],
        "summary_scalar_abs_diff_max": summary_details["scalar_abs_diff_max"],
        "summary_scalar_rel_diff_max": summary_details["scalar_rel_diff_max"],
        "probe_value_max_abs_diff": summary_details["probe_value_max_abs_diff"],
        "cpp_diagonal_max_abs": float(cpp.get("diagonal_max_abs", 0.0)),
        "python_diagonal_max_abs": float(py.get("diagonal_max_abs", 0.0)),
        "cpp_symmetry_max_abs": float(cpp.get("symmetry_max_abs", 0.0)),
        "python_symmetry_max_abs": float(py.get("symmetry_max_abs", 0.0)),
    }
    if stage_key == "distance" and core_distance_details is not None:
        result["core_distance_summary_scalar_diffs"] = core_distance_details["scalar_diffs"]
        result["core_distance_probe_value_max_abs_diff"] = core_distance_details["probe_value_max_abs_diff"]
        result["core_distance_hash_equal"] = core_distance_details["hash_equal"]
    if stage_key in {"select", "full"}:
        result["cpp_noise_count"] = int(cpp.get("noise_count", -1))
        result["python_noise_count"] = int(py.get("noise_count", -1))
        result["cpp_cluster_count"] = int(cpp.get("cluster_count", -1))
        result["python_cluster_count"] = int(py.get("cluster_count", -1))
        if label_details is not None:
            result["label_summary_scalar_diffs"] = label_details["scalar_diffs"]
            result["label_summary_scalar_abs_diff_max"] = label_details["scalar_abs_diff_max"]
            result["label_summary_scalar_rel_diff_max"] = label_details["scalar_rel_diff_max"]
            result["label_hash_equal"] = label_details["hash_equal"]
            result["label_probe_value_max_abs_diff"] = label_details["probe_value_max_abs_diff"]
        if probability_details is not None:
            result["probability_summary_scalar_diffs"] = probability_details["scalar_diffs"]
            result["probability_summary_scalar_abs_diff_max"] = probability_details["scalar_abs_diff_max"]
            result["probability_summary_scalar_rel_diff_max"] = probability_details["scalar_rel_diff_max"]
            result["probability_hash_equal"] = probability_details["hash_equal"]
            result["probability_probe_value_max_abs_diff"] = probability_details["probe_value_max_abs_diff"]
    return result
