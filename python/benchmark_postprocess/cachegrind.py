from pathlib import Path
from typing import Any

from benchmark_postprocess.io import load_json
from benchmark_pipeline.cachegrind import (
    CACHEGRIND_MANIFEST_FILENAME,
    CACHEGRIND_PREFIX,
    CACHEGRIND_SCHEMA_VERSION,
)
from benchmark_pipeline.paths import repo_relative_path


def _empty_manifest(results_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": "No Cachegrind manifest was found.",
        "enabled": False,
        "results_dir": str(results_dir),
        "cache_model": {"I1": None, "D1": None, "LL": None},
        "planned_record_count": 0,
        "exclusion_count": 0,
        "planned_records": [],
        "exclusions": [],
    }


def load_cachegrind_manifest(results_dir: str | Path) -> dict[str, Any]:
    results_dir = repo_relative_path(results_dir)
    path = results_dir / CACHEGRIND_MANIFEST_FILENAME

    if not path.exists():
        return _empty_manifest(results_dir)

    manifest = load_json(path)
    if not isinstance(manifest, dict) or "planned_records" not in manifest:
        raise ValueError(f"Invalid Cachegrind manifest: {path}")

    return manifest


def load_cachegrind_records(results_dir: str | Path) -> list[dict[str, Any]]:
    results_dir = repo_relative_path(results_dir)
    records: list[dict[str, Any]] = []

    for path in sorted(results_dir.glob(f"{CACHEGRIND_PREFIX}.*.json")):
        payload = load_json(path)
        if int(payload.get("schema_version", 0)) != CACHEGRIND_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Cachegrind record schema in {path}")
        records.append(payload)

    return sorted(
        records,
        key=lambda item: (
            int(item["D"]),
            int(item["N"]),
            int(item["K"]),
            str(item["stage_key"]),
            str(item["cpp_case"]),
            str(item["params_key"]),
        ),
    )


def build_cachegrind_summary(results_dir: str | Path) -> dict[str, Any]:
    results_dir = repo_relative_path(results_dir)
    manifest = load_cachegrind_manifest(results_dir)
    enabled = bool(manifest.get("enabled", False))
    records = load_cachegrind_records(results_dir) if enabled else []
    planned_record_count = int(manifest.get("planned_record_count", 0))

    return {
        "enabled": enabled,
        "source": str(results_dir / CACHEGRIND_MANIFEST_FILENAME),
        "results_dir": manifest.get("results_dir", str(results_dir)),
        "cache_model": manifest.get(
            "cache_model",
            {"I1": None, "D1": None, "LL": None},
        ),
        "record_count": len(records),
        "planned_record_count": planned_record_count,
        "missing_record_count": max(0, planned_record_count - len(records)),
        "exclusion_count": int(manifest.get("exclusion_count", 0)),
        "exclusions": manifest.get("exclusions", []),
        "planned_records": manifest.get("planned_records", []),
        "records": records,
    }
