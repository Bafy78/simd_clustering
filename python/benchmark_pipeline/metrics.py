import json
from collections.abc import Sequence


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, payload: dict) -> None:
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


def validate_gmm_process_metrics(candidate: dict, reference: dict, path: str) -> None:
    if candidate.get("algorithm") != reference.get("algorithm"):
        raise RuntimeError(f"algorithm mismatch in {path}")

    if candidate.get("covariance_type") != reference.get("covariance_type"):
        raise RuntimeError(f"covariance_type mismatch in {path}")

    if int(candidate["iterations"]) != int(reference["iterations"]):
        raise RuntimeError(
            f"iteration mismatch in {path}: "
            f"{candidate['iterations']} vs {reference['iterations']}"
        )

    if bool(candidate["converged"]) != bool(reference["converged"]):
        raise RuntimeError(f"converged mismatch in {path}")

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


def validate_cpp_process_metrics(process_metrics: list[str]) -> dict:
    if not process_metrics:
        raise RuntimeError("No C++ process metrics files to validate")

    reference = load_json(process_metrics[0])

    for path in process_metrics[1:]:
        candidate = load_json(path)

        if candidate.get("schema_version") != reference.get("schema_version"):
            raise RuntimeError(f"schema_version mismatch in {path}")

        if candidate.get("language") != reference.get("language"):
            raise RuntimeError(f"language mismatch in {path}")

        if reference.get("algorithm") == "gmm":
            validate_gmm_process_metrics(candidate, reference, path)
            continue

        if int(candidate["iterations"]) != int(reference["iterations"]):
            raise RuntimeError(
                f"iteration mismatch in {path}: "
                f"{candidate['iterations']} vs {reference['iterations']}"
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

    reference["process_metrics_verified"] = True
    reference["process_metrics_count"] = len(process_metrics)

    return reference


def compute_lloyd_parity(
    *,
    config_id: str,
    cpp_metrics_file: str,
    py_metrics_file: str,
    output_file: str,
    tolerance_pct: float,
) -> dict:
    cpp = load_json(cpp_metrics_file)
    py = load_json(py_metrics_file)

    cpp_inertia = float(cpp["inertia"])
    py_inertia = float(py["inertia"])

    inertia_diff_abs = abs(cpp_inertia - py_inertia)
    scale = max(abs(cpp_inertia), abs(py_inertia))
    if scale > 0.0:
        inertia_diff_pct = inertia_diff_abs / scale * 100.0
    else:
        inertia_diff_pct = 0.0

    parity = {
        "schema_version": 1,
        "config_id": config_id,
        "cpp_iterations": int(cpp["iterations"]),
        "python_iterations": int(py["iterations"]),
        "cpp_inertia": cpp_inertia,
        "python_inertia": py_inertia,
        "inertia_diff_abs": inertia_diff_abs,
        "inertia_diff_pct": inertia_diff_pct,
        "tolerance_pct": tolerance_pct,
        "status": "PASS" if inertia_diff_pct <= tolerance_pct else "FAIL",
        "cpp_cluster_counts": cpp.get("cluster_counts"),
        "python_cluster_counts": py.get("cluster_counts"),
        "cpp_cluster_inertia": cpp.get("cluster_inertia"),
        "python_cluster_inertia": py.get("cluster_inertia"),
    }

    write_json(output_file, parity)

    if parity["status"] != "PASS":
        print(
            f"WARNING: Lloyd parity failed for {config_id}: "
            f"{inertia_diff_pct:.12g}% > {tolerance_pct}%"
        )

    return parity
