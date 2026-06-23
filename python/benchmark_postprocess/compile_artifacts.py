from pathlib import Path
from typing import Any

from benchmark_metadata import stage_display_name
from benchmark_postprocess.io import load_json
from benchmark_pipeline.compile_artifacts import COMPILE_ARTIFACTS_FILENAME


CompileIdentity = tuple[int, str, str, str]


def _compile_identity_from_record(record: dict[str, Any]) -> CompileIdentity:
    return (
        int(record["dimensions"]),
        str(record["phase_key"]),
        str(record["stage_key"]),
        str(record["variant_key"]),
    )


def _artifact_stage_keys(record: dict[str, Any]) -> tuple[str, ...]:
    """Return the benchmark stages covered by a compile artifact record.

    Older artifacts stored a single ``stage_key``. Newer artifacts store
    ``stage_keys`` because one compiled binary can serve several benchmark
    stages. Postprocessing still expands them to one row per concrete stage so
    downstream reporting can keep using ``stage_key``.
    """
    if "stage_key" in record:
        return (str(record["stage_key"]),)

    stage_keys = record.get("stage_keys")
    if not isinstance(stage_keys, list) or not stage_keys:
        raise KeyError("stage_key")

    return tuple(str(stage_key) for stage_key in stage_keys)


def _compile_identities_from_artifact(record: dict[str, Any]) -> tuple[CompileIdentity, ...]:
    return tuple(
        (
            int(record["D"]),
            str(record["phase_key"]),
            stage_key,
            str(record["variant_key"]),
        )
        for stage_key in _artifact_stage_keys(record)
    )


def _artifact_record_for_stage(record: dict[str, Any], stage_key: str) -> dict[str, Any]:
    expanded = dict(record)
    expanded["stage_key"] = stage_key
    expanded["stage"] = stage_display_name(stage_key)
    return expanded


def _artifact_sort_key(record: dict[str, Any]) -> tuple[int, str, str, str, str]:
    return (
        int(record["D"]),
        str(record["phase_key"]),
        str(record["stage_key"]),
        str(record["variant_key"]),
        str(record["cpp_case"]),
    )


def build_compile_artifact_summary(
    records: list[dict[str, Any]],
    *,
    data_dir: Path,
) -> dict[str, Any]:
    expected_identities = {
        _compile_identity_from_record(record)
        for record in records
        if record.get("language_key") == "cpp"
    }

    artifact_path = data_dir / COMPILE_ARTIFACTS_FILENAME
    if not expected_identities:
        return {
            "enabled": False,
            "schema_version": 1,
            "artifact_path": str(artifact_path),
            "records": [],
        }

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Expected C++ compile artifact report at {artifact_path}. "
            "Run the benchmark orchestrator again so compile-time executable "
            "sizes and resolved architectures are captured."
        )

    payload = load_json(artifact_path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported compile artifact schema in {artifact_path}")

    indexed: dict[CompileIdentity, dict[str, Any]] = {}
    for artifact_record in payload.get("records", []):
        for identity in _compile_identities_from_artifact(artifact_record):
            if identity not in expected_identities:
                continue
            if identity in indexed:
                raise ValueError(
                    "Duplicate compile artifact for "
                    f"D={identity[0]}, phase={identity[1]!r}, "
                    f"stage={identity[2]!r}, variant={identity[3]!r}"
                )
            indexed[identity] = _artifact_record_for_stage(
                artifact_record,
                stage_key=identity[2],
            )

    missing = sorted(expected_identities - set(indexed))
    if missing:
        missing_text = ", ".join(
            f"D={D}/phase={phase_key}/stage={stage_key}/variant={variant_key}"
            for D, phase_key, stage_key, variant_key in missing
        )
        raise RuntimeError(f"Missing compile artifact records for: {missing_text}")

    rows = [indexed[key] for key in sorted(indexed)]
    architectures = sorted({str(row["architecture"]) for row in rows})

    return {
        "enabled": True,
        "schema_version": 1,
        "artifact_path": str(artifact_path),
        "record_count": len(rows),
        "architectures": architectures,
        "records": sorted(rows, key=_artifact_sort_key),
    }
