from pathlib import Path
from typing import Any

from benchmark_postprocess.io import load_json
from benchmark_postprocess.naming import LLOYD_PARITY_JSON_RE


def lloyd_iteration_count(
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
        return int(parity["cpp_iterations"])

    if lang_key == "py":
        return int(parity["python_iterations"])

    raise RuntimeError(f"Unexpected Lloyd language key: {lang_key!r}")


def normalize_lloyd_parity_record(
    record: dict[str, Any],
    *,
    config_id: str,
) -> dict[str, Any]:
    required_fields = [
        "cpp_iterations",
        "python_iterations",
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
        "cpp_iterations": int(record["cpp_iterations"]),
        "python_iterations": int(record["python_iterations"]),
        "cpp_inertia": float(record["cpp_inertia"]),
        "python_inertia": float(record["python_inertia"]),
        "inertia_diff_abs": float(record["inertia_diff_abs"]),
        "inertia_diff_pct": float(record["inertia_diff_pct"]),
        "tolerance_pct": float(record["tolerance_pct"]),
        "status": str(record["status"]),
    }


def load_lloyd_parity_map(data_dir: Path) -> dict[str, dict[str, Any]]:
    parity_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for path in sorted(data_dir.glob("lloyd_parity_*.json")):
        match = LLOYD_PARITY_JSON_RE.match(path.name)

        if not match:
            print(f"Skipping malformed Lloyd parity filename: {path.name}")
            continue

        dim = int(match.group("dim"))
        samples = int(match.group("samples"))
        clusters = int(match.group("clusters"))
        config_id = f"{dim}D_{samples}S_{clusters}K"

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
