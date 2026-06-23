import json
from collections.abc import Sequence
from pathlib import Path


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, payload: dict) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def assert_close(name: str, a: float, b: float, *, rel_tol=1e-10, abs_tol=1e-8):
    if a == b:
        return

    scale = max(abs(a), abs(b), 1.0)
    diff = abs(a - b)

    if diff > max(abs_tol, rel_tol * scale):
        raise RuntimeError(
            f"C++ metrics mismatch for {name}: {a} vs {b} " f"(abs diff={diff})"
        )


def _assert_sequence_close(name: str, candidate, reference) -> None:
    if len(candidate) != len(reference):
        raise RuntimeError(
            f"C++ metrics mismatch for {name}: length {len(candidate)} vs {len(reference)}"
        )

    for i, (a, b) in enumerate(zip(candidate, reference)):
        assert_close(f"{name}[{i}]", float(a), float(b))


def _assert_matrix_close(name: str, candidate, reference) -> None:
    if len(candidate) != len(reference):
        raise RuntimeError(
            f"C++ metrics mismatch for {name}: row count {len(candidate)} vs {len(reference)}"
        )

    for row, (candidate_row, reference_row) in enumerate(zip(candidate, reference)):
        if len(candidate_row) != len(reference_row):
            raise RuntimeError(
                f"C++ metrics mismatch for {name}[{row}]: "
                f"length {len(candidate_row)} vs {len(reference_row)}"
            )

        for col, (a, b) in enumerate(zip(candidate_row, reference_row)):
            assert_close(f"{name}[{row}][{col}]", float(a), float(b))


def _is_non_string_sequence(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _shape_of(value) -> tuple[int, ...]:
    if not _is_non_string_sequence(value):
        return ()

    child_shapes = [_shape_of(child) for child in value]
    if not child_shapes:
        return (0,)

    first = child_shapes[0]
    for child_shape in child_shapes[1:]:
        if child_shape != first:
            raise RuntimeError("ragged array")

    return (len(value), *first)


def _assert_array_close(name: str, candidate, reference) -> None:
    try:
        candidate_shape = _shape_of(candidate)
        reference_shape = _shape_of(reference)
    except RuntimeError as exc:
        raise RuntimeError(f"C++ metrics mismatch for {name}: {exc}") from exc

    if candidate_shape != reference_shape:
        raise RuntimeError(
            f"C++ metrics mismatch for {name}: "
            f"shape {candidate_shape} vs {reference_shape}"
        )

    def visit(candidate_value, reference_value, suffix: str) -> None:
        if not _is_non_string_sequence(candidate_value):
            assert_close(
                f"{name}{suffix}", float(candidate_value), float(reference_value)
            )
            return

        for i, (candidate_child, reference_child) in enumerate(
            zip(candidate_value, reference_value)
        ):
            visit(candidate_child, reference_child, f"{suffix}[{i}]")

    visit(candidate, reference, "")


def validate_gmm_timing_process_metrics(
    candidate: dict, reference: dict, path: str
) -> None:
    if candidate.get("phase") != reference.get("phase"):
        raise RuntimeError(f"phase mismatch in {path}")

    if candidate.get("covariance_type") != reference.get("covariance_type"):
        raise RuntimeError(f"covariance_type mismatch in {path}")

    if int(candidate["algorithm_iterations"]) != int(reference["algorithm_iterations"]):
        raise RuntimeError(
            f"algorithm-iteration mismatch in {path}: "
            f"{candidate['algorithm_iterations']} vs {reference['algorithm_iterations']}"
        )

    assert_close(
        f"lower_bound in {path}",
        float(candidate["lower_bound"]),
        float(reference["lower_bound"]),
    )

    _assert_sequence_close(
        f"lower_bounds in {path}",
        candidate.get("lower_bounds", []),
        reference.get("lower_bounds", []),
    )
    _assert_sequence_close(
        f"weights in {path}",
        candidate.get("weights", []),
        reference.get("weights", []),
    )
    _assert_array_close(
        f"covariances in {path}",
        candidate.get("covariances", []),
        reference.get("covariances", []),
    )
    _assert_matrix_close(
        f"means in {path}",
        candidate.get("means", []),
        reference.get("means", []),
    )


def validate_hdbscan_timing_process_metrics(candidate: dict, reference: dict, path: str) -> None:
    if candidate.get("phase") != reference.get("phase"):
        raise RuntimeError(f"phase mismatch in {path}")

    if candidate.get("stage") != reference.get("stage"):
        raise RuntimeError(f"stage mismatch in {path}")

    if candidate.get("dtype") != reference.get("dtype"):
        raise RuntimeError(f"dtype mismatch in {path}")

    if candidate.get("shape") != reference.get("shape"):
        raise RuntimeError(f"shape mismatch in {path}")

    if int(candidate.get("min_samples", -1)) != int(reference.get("min_samples", -1)):
        raise RuntimeError(f"min_samples mismatch in {path}")

    assert_close(
        f"diagonal_max_abs in {path}",
        float(candidate.get("diagonal_max_abs", 0.0)),
        float(reference.get("diagonal_max_abs", 0.0)),
    )
    assert_close(
        f"symmetry_max_abs in {path}",
        float(candidate.get("symmetry_max_abs", 0.0)),
        float(reference.get("symmetry_max_abs", 0.0)),
    )

    candidate_summary = candidate.get("summary", {})
    reference_summary = reference.get("summary", {})

    for field in (
        "value_count",
        "finite_count",
        "nan_count",
        "pos_inf_count",
        "neg_inf_count",
        "fnv1a64_float32",
    ):
        if candidate_summary.get(field) != reference_summary.get(field):
            raise RuntimeError(
                f"HDBSCAN summary {field} mismatch in {path}: "
                f"{candidate_summary.get(field)!r} vs {reference_summary.get(field)!r}"
            )

    for field in ("sum", "sum_abs", "sum_squares", "weighted_sum", "min", "max"):
        assert_close(
            f"summary.{field} in {path}",
            float(candidate_summary[field]),
            float(reference_summary[field]),
            rel_tol=1e-7,
            abs_tol=1e-6,
        )

    candidate_probes = candidate_summary.get("probes", [])
    reference_probes = reference_summary.get("probes", [])
    if len(candidate_probes) != len(reference_probes):
        raise RuntimeError(f"HDBSCAN probe count mismatch in {path}")

    for i, (cand, ref) in enumerate(zip(candidate_probes, reference_probes)):
        if cand.get("index") != ref.get("index"):
            raise RuntimeError(f"HDBSCAN probe index mismatch in {path} at probe {i}")
        assert_close(
            f"summary.probes[{i}].value in {path}",
            float(cand["value"]),
            float(ref["value"]),
            rel_tol=1e-7,
            abs_tol=1e-6,
        )


def validate_cpp_timing_process_metrics(timing_process_metrics: list[str]) -> dict:
    if not timing_process_metrics:
        raise RuntimeError("No C++ timing-process metrics files to validate")

    reference = load_json(timing_process_metrics[0])

    for path in timing_process_metrics[1:]:
        candidate = load_json(path)

        if candidate.get("schema_version") != reference.get("schema_version"):
            raise RuntimeError(f"schema_version mismatch in {path}")

        if candidate.get("language") != reference.get("language"):
            raise RuntimeError(f"language mismatch in {path}")

        if reference.get("phase") == "gmm":
            validate_gmm_timing_process_metrics(candidate, reference, path)
            continue

        if reference.get("phase") == "hdbscan":
            validate_hdbscan_timing_process_metrics(candidate, reference, path)
            continue

        if int(candidate["algorithm_iterations"]) != int(
            reference["algorithm_iterations"]
        ):
            raise RuntimeError(
                f"algorithm-iteration mismatch in {path}: "
                f"{candidate['algorithm_iterations']} vs {reference['algorithm_iterations']}"
            )

        if candidate["cluster_counts"] != reference["cluster_counts"]:
            raise RuntimeError(f"cluster_counts mismatch in {path}")

        assert_close(
            f"inertia in {path}",
            float(candidate["inertia"]),
            float(reference["inertia"]),
        )

        if len(candidate["cluster_inertia"]) != len(reference["cluster_inertia"]):
            raise RuntimeError(f"cluster_inertia length mismatch in {path}")

        if len(candidate.get("centroids", [])) != len(reference.get("centroids", [])):
            raise RuntimeError(f"centroid count mismatch in {path}")

        for i, (a, b) in enumerate(
            zip(candidate["cluster_inertia"], reference["cluster_inertia"])
        ):
            assert_close(
                f"cluster_inertia[{i}] in {path}",
                float(a),
                float(b),
            )

        for k, (cand_centroid, ref_centroid) in enumerate(
            zip(candidate.get("centroids", []), reference.get("centroids", []))
        ):
            if len(cand_centroid) != len(ref_centroid):
                raise RuntimeError(f"centroid[{k}] dimensionality mismatch in {path}")

            for d, (a, b) in enumerate(zip(cand_centroid, ref_centroid)):
                assert_close(
                    f"centroid[{k}][{d}] in {path}",
                    float(a),
                    float(b),
                )

    reference["timing_process_metrics_verified"] = True
    reference["timing_process_metrics_count"] = len(timing_process_metrics)

    return reference
