import json


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
