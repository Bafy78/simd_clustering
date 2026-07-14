#!/usr/bin/env python3
"""Merge two post-processed benchmark summary JSON files.

The merge is intentionally hierarchical: data is preserved unless the two
summaries contain the same configuration, phase, stage, variant, and parameterization.
When that full identity is present in both inputs, the high-priority summary's
parameterization block replaces the low-priority block atomically.

Examples
--------
python python/concatenate_benchmark_summaries.py \
  --low-priority datasets/old/benchmark_summary.json \
  --high-priority datasets/new/benchmark_summary.json \
  --output datasets/benchmark_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

JsonDict = dict[str, Any]
T = TypeVar("T")

from benchmark_metadata import DEFAULT_DATASET_KEY


@dataclass
class MergeStats:
    configs_added: int = 0
    phases_added: int = 0
    stages_added: int = 0
    variants_added: int = 0
    parameterizations_added: int = 0
    parameterizations_replaced: int = 0
    exclusions_kept: int = 0
    exclusions_dropped_due_to_data: int = 0
    compile_artifact_records: int = 0
    cachegrind_records: int = 0
    cachegrind_planned_records: int = 0
    cachegrind_exclusions: int = 0
    spill_detection_records: int = 0
    spill_detection_scan_targets: int = 0
    spill_detection_errors: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two benchmark_summary.json files while preserving all "
            "non-redundant config/phase/stage/variant/parameterization data."
        )
    )
    parser.add_argument(
        "--low-priority",
        type=Path,
        required=True,
        help="Base summary. Data from this file is kept unless overridden.",
    )
    parser.add_argument(
        "--high-priority",
        type=Path,
        required=True,
        help="Override summary. Exact duplicate parameterizations win.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the merged benchmark summary JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a JSON object.")

    return data


def write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _required_int(entry: JsonDict, field_name: str, context: str) -> int:
    if field_name not in entry:
        raise ValueError(f"Missing {field_name!r} in {context}: {entry!r}")

    try:
        return int(entry[field_name])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer-compatible {field_name!r} in {context}: {entry!r}"
        ) from exc


def config_key(config: JsonDict) -> tuple[str, int, int, int]:
    return (
        str(config.get("dataset", DEFAULT_DATASET_KEY)),
        _required_int(config, "dimensions", "config"),
        _required_int(config, "samples", "config"),
        _required_int(config, "clusters", "config"),
    )


def config_id_for_key(key: tuple[str, int, int, int]) -> str:
    dataset, dimensions, samples, clusters = key
    return f"{dataset}_{dimensions}D_{samples}N_{clusters}K"


def phase_entry_key(mapping_key: str, phase: JsonDict) -> str:
    return str(phase.get("phase_key", mapping_key))


def stage_entry_key(mapping_key: str, stage: JsonDict) -> str:
    return str(stage["stage_key"])


def variant_entry_key(mapping_key: str, variant: JsonDict) -> str:
    return str(variant.get("variant_key", mapping_key))


def parameterization_entry_key(mapping_key: str, parameterization: JsonDict) -> str:
    return str(parameterization.get("params_key", mapping_key))


def exclusion_key(exclusion: JsonDict) -> tuple[str, int, int, int, str, str]:
    return (
        str(exclusion.get("dataset", DEFAULT_DATASET_KEY)),
        _required_int(exclusion, "dimensions", "exclusion"),
        _required_int(exclusion, "samples", "exclusion"),
        _required_int(exclusion, "clusters", "exclusion"),
        str(exclusion.get("phase_key", exclusion.get("phase", ""))),
        str(exclusion["stage_key"]),
    )


def _sorted_dict_items_by_key(
    values: dict[str, JsonDict],
    key_fn: Callable[[str, JsonDict], Any],
) -> Iterable[tuple[str, JsonDict]]:
    return sorted(values.items(), key=lambda item: (key_fn(item[0], item[1]), item[0]))


def _replace_mapping_entry(
    mapping: dict[str, JsonDict],
    old_display_key: str,
    new_display_key: str,
    new_value: JsonDict,
) -> None:
    if old_display_key != new_display_key:
        del mapping[old_display_key]
    mapping[new_display_key] = new_value


def _index_nested_mapping(
    mapping: Any,
    key_fn: Callable[[str, JsonDict], str],
    context: str,
) -> dict[str, str]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Expected {context} to be a JSON object, got {type(mapping)}")

    index: dict[str, str] = {}
    for display_key, entry in mapping.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected {context}[{display_key!r}] to be a JSON object, "
                f"got {type(entry)}"
            )

        stable_key = key_fn(str(display_key), entry)
        if stable_key in index:
            raise ValueError(
                f"Duplicate stable key {stable_key!r} found in {context}. "
                f"Display keys: {index[stable_key]!r}, {display_key!r}"
            )
        index[stable_key] = str(display_key)

    return index


def _sorted_nested_mapping(
    mapping: dict[str, JsonDict], key_fn: Callable[[str, JsonDict], str]
) -> dict[str, JsonDict]:
    return {
        display_key: entry
        for display_key, entry in _sorted_dict_items_by_key(mapping, key_fn)
    }


def _copy_non_container_fields(dst: JsonDict, src: JsonDict, *, skip: set[str]) -> JsonDict:
    """Copy high-priority scalar/metadata fields without touching child containers."""
    merged = copy.deepcopy(dst)
    for key, value in src.items():
        if key in skip:
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def merge_parameterization(low: JsonDict, high: JsonDict, stats: MergeStats) -> JsonDict:
    # Keep this as the atomic replacement unit. Fields such as languages,
    # speedup, and parity are derived together and should not be interleaved
    # between summaries.
    stats.parameterizations_replaced += 1
    return copy.deepcopy(high)


def merge_variant(low: JsonDict, high: JsonDict, stats: MergeStats) -> JsonDict:
    merged = _copy_non_container_fields(low, high, skip={"parameterizations"})

    low_parameterizations = copy.deepcopy(low.get("parameterizations", {}))
    high_parameterizations = high.get("parameterizations", {})
    low_index = _index_nested_mapping(
        low_parameterizations,
        parameterization_entry_key,
        "variant.parameterizations",
    )
    high_index = _index_nested_mapping(
        high_parameterizations,
        parameterization_entry_key,
        "variant.parameterizations",
    )

    for params_stable_key, high_display_key in high_index.items():
        high_parameterization = high_parameterizations[high_display_key]
        if params_stable_key in low_index:
            low_display_key = low_index[params_stable_key]
            merged_parameterization = merge_parameterization(
                low_parameterizations[low_display_key], high_parameterization, stats
            )
            _replace_mapping_entry(
                low_parameterizations,
                low_display_key,
                high_display_key,
                merged_parameterization,
            )
            low_index[params_stable_key] = high_display_key
        else:
            low_parameterizations[high_display_key] = copy.deepcopy(high_parameterization)
            low_index[params_stable_key] = high_display_key
            stats.parameterizations_added += 1

    merged["parameterizations"] = _sorted_nested_mapping(
        low_parameterizations, parameterization_entry_key
    )
    return merged


def _phase_stage_mapping(phase: JsonDict) -> dict[str, JsonDict]:
    stages = phase.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError(f"Expected phase.stages to be a JSON object, got {type(stages)}")
    return copy.deepcopy(stages)


def merge_stage(low: JsonDict, high: JsonDict, stats: MergeStats) -> JsonDict:
    merged = _copy_non_container_fields(low, high, skip={"variants"})

    low_variants = copy.deepcopy(low.get("variants", {}))
    high_variants = high.get("variants", {})
    low_index = _index_nested_mapping(low_variants, variant_entry_key, "stage.variants")
    high_index = _index_nested_mapping(high_variants, variant_entry_key, "stage.variants")

    for variant_stable_key, high_display_key in high_index.items():
        high_variant = high_variants[high_display_key]
        if variant_stable_key in low_index:
            low_display_key = low_index[variant_stable_key]
            merged_variant = merge_variant(low_variants[low_display_key], high_variant, stats)
            _replace_mapping_entry(
                low_variants,
                low_display_key,
                high_display_key,
                merged_variant,
            )
            low_index[variant_stable_key] = high_display_key
        else:
            low_variants[high_display_key] = copy.deepcopy(high_variant)
            low_index[variant_stable_key] = high_display_key
            stats.variants_added += 1

    merged["variants"] = _sorted_nested_mapping(low_variants, variant_entry_key)
    return merged


def merge_phase(low: JsonDict, high: JsonDict, stats: MergeStats) -> JsonDict:
    merged = _copy_non_container_fields(low, high, skip={"variants", "stages"})

    low_stages = _phase_stage_mapping(low)
    high_stages = _phase_stage_mapping(high)
    low_index = _index_nested_mapping(low_stages, stage_entry_key, "phase.stages")
    high_index = _index_nested_mapping(high_stages, stage_entry_key, "phase.stages")

    for stage_stable_key, high_display_key in high_index.items():
        high_stage = high_stages[high_display_key]
        if stage_stable_key in low_index:
            low_display_key = low_index[stage_stable_key]
            merged_stage = merge_stage(low_stages[low_display_key], high_stage, stats)
            _replace_mapping_entry(
                low_stages,
                low_display_key,
                high_display_key,
                merged_stage,
            )
            low_index[stage_stable_key] = high_display_key
        else:
            low_stages[high_display_key] = copy.deepcopy(high_stage)
            low_index[stage_stable_key] = high_display_key
            stats.stages_added += 1

    merged["stages"] = _sorted_nested_mapping(low_stages, stage_entry_key)
    return merged


def merge_config(low: JsonDict, high: JsonDict, stats: MergeStats) -> JsonDict:
    merged = _copy_non_container_fields(low, high, skip={"phases", "excluded_phases"})

    low_phases = copy.deepcopy(low.get("phases", {}))
    high_phases = high.get("phases", {})
    low_index = _index_nested_mapping(low_phases, phase_entry_key, "config.phases")
    high_index = _index_nested_mapping(high_phases, phase_entry_key, "config.phases")

    for phase_stable_key, high_display_key in high_index.items():
        high_phase = high_phases[high_display_key]
        if phase_stable_key in low_index:
            low_display_key = low_index[phase_stable_key]
            merged_phase = merge_phase(low_phases[low_display_key], high_phase, stats)
            _replace_mapping_entry(
                low_phases,
                low_display_key,
                high_display_key,
                merged_phase,
            )
            low_index[phase_stable_key] = high_display_key
        else:
            low_phases[high_display_key] = copy.deepcopy(high_phase)
            low_index[phase_stable_key] = high_display_key
            stats.phases_added += 1

    merged["phases"] = _sorted_nested_mapping(low_phases, phase_entry_key)
    merged["excluded_phases"] = {}
    return merged


def index_configs(configs: Any, context: str) -> dict[tuple[str, int, int, int], JsonDict]:
    if configs is None:
        return {}
    if not isinstance(configs, list):
        raise ValueError(f"Expected {context} to be a list, got {type(configs)}")

    indexed: dict[tuple[str, int, int, int], JsonDict] = {}
    for config in configs:
        if not isinstance(config, dict):
            raise ValueError(f"Expected entries in {context} to be objects, got {type(config)}")
        key = config_key(config)
        if key in indexed:
            raise ValueError(f"Duplicate config key {key!r} found in {context}")
        normalized = copy.deepcopy(config)
        normalized["dataset"] = key[0]
        normalized["config_id"] = config_id_for_key(key)
        indexed[key] = normalized

    return indexed


def concrete_phase_stage_keys(configs: Iterable[JsonDict]) -> set[tuple[str, int, int, int, str, str]]:
    keys: set[tuple[str, int, int, int, str, str]] = set()
    for config in configs:
        cfg_key = config_key(config)
        phases = config.get("phases", {})
        if not isinstance(phases, dict):
            raise ValueError(f"Expected phases to be a JSON object in config {cfg_key!r}")
        for display_key, phase in phases.items():
            if not isinstance(phase, dict):
                raise ValueError(
                    f"Expected phase {display_key!r} in config {cfg_key!r} to be an object"
                )
            phase_key = phase_entry_key(str(display_key), phase)
            stages = _phase_stage_mapping(phase)
            for stage_display_key, stage in stages.items():
                keys.add((*cfg_key, phase_key, stage_entry_key(str(stage_display_key), stage)))
    return keys


def collect_exclusions(summary: JsonDict) -> list[JsonDict]:
    """Collect top-level exclusions."""
    collected: dict[tuple[str, int, int, int, str, str], JsonDict] = {}

    for exclusion in summary.get("exclusions", []) or []:
        if not isinstance(exclusion, dict):
            raise ValueError(f"Expected exclusion entries to be objects: {exclusion!r}")
        collected[exclusion_key(exclusion)] = copy.deepcopy(exclusion)

    return [collected[key] for key in sorted(collected)]


def merge_exclusions(
    low: JsonDict,
    high: JsonDict,
    configs: dict[tuple[str, int, int, int], JsonDict],
    stats: MergeStats,
) -> list[JsonDict]:
    merged_by_key: dict[tuple[str, int, int, int, str, str], JsonDict] = {}

    for exclusion in collect_exclusions(low):
        merged_by_key[exclusion_key(exclusion)] = copy.deepcopy(exclusion)
    for exclusion in collect_exclusions(high):
        merged_by_key[exclusion_key(exclusion)] = copy.deepcopy(exclusion)

    phase_data_keys = concrete_phase_stage_keys(configs.values())
    active: list[JsonDict] = []
    for key in sorted(merged_by_key):
        if key in phase_data_keys:
            stats.exclusions_dropped_due_to_data += 1
            continue
        active.append(merged_by_key[key])

    stats.exclusions_kept = len(active)
    return active


def compile_artifact_key(record: JsonDict) -> tuple[int, str, str, str, str]:
    return (
        _required_int(record, "D", "compile artifact"),
        str(record.get("phase_key", "")),
        str(record["stage_key"]),
        str(record.get("variant_key", "")),
        str(record.get("cpp_case", "")),
    )


def merge_compile_artifacts(
    low: JsonDict,
    high: JsonDict,
    stats: MergeStats,
) -> JsonDict:
    low_artifacts = low.get("compile_artifacts", {}) or {}
    high_artifacts = high.get("compile_artifacts", {}) or {}

    if not isinstance(low_artifacts, dict):
        raise ValueError("Expected low-priority compile_artifacts to be a JSON object")
    if not isinstance(high_artifacts, dict):
        raise ValueError("Expected high-priority compile_artifacts to be a JSON object")

    merged = copy.deepcopy(low_artifacts)
    merged.update(
        {
            key: copy.deepcopy(value)
            for key, value in high_artifacts.items()
            if key != "records"
        }
    )
    merged.setdefault("schema_version", 1)

    records_by_key: dict[tuple[int, str, str, str, str], JsonDict] = {}
    for record in low_artifacts.get("records", []) or []:
        records_by_key[compile_artifact_key(record)] = copy.deepcopy(record)
    for record in high_artifacts.get("records", []) or []:
        records_by_key[compile_artifact_key(record)] = copy.deepcopy(record)

    records = [records_by_key[key] for key in sorted(records_by_key)]
    merged["records"] = records
    merged["record_count"] = len(records)
    merged["architectures"] = sorted(
        {str(record["architecture"]) for record in records}
    )
    merged["enabled"] = bool(records)
    stats.compile_artifact_records = len(records)
    return merged



def cachegrind_record_key(record: JsonDict) -> tuple[str, int, int, int, str, str, str]:
    return (
        str(record.get("dataset", DEFAULT_DATASET_KEY)),
        _required_int(record, "D", "Cachegrind record"),
        _required_int(record, "N", "Cachegrind record"),
        _required_int(record, "K", "Cachegrind record"),
        str(record["stage_key"]),
        str(record.get("cpp_case", "")),
        str(record.get("params_key", "default")),
    )


def cachegrind_target_key(record: JsonDict) -> tuple[str, int, int, int, str, str, str]:
    return (
        str(record.get("dataset", DEFAULT_DATASET_KEY)),
        _required_int(record, "D", "Cachegrind planned record"),
        _required_int(record, "N", "Cachegrind planned record"),
        _required_int(record, "K", "Cachegrind planned record"),
        str(record["stage_key"]),
        str(record.get("cpp_case", "")),
        str(record.get("params_key", "default")),
    )


def cachegrind_exclusion_key(exclusion: JsonDict) -> tuple[str, int, int, int, str, str, str]:
    return (
        str(exclusion.get("dataset", DEFAULT_DATASET_KEY)),
        _required_int(exclusion, "dimensions", "Cachegrind exclusion"),
        _required_int(exclusion, "samples", "Cachegrind exclusion"),
        _required_int(exclusion, "clusters", "Cachegrind exclusion"),
        str(exclusion["stage_key"]),
        str(exclusion.get("cpp_case", "")),
        str(exclusion.get("params_key", "default")),
    )


def merge_cachegrind_summaries(
    low: JsonDict,
    high: JsonDict,
    stats: MergeStats,
) -> JsonDict:
    low_cachegrind = low.get("cachegrind", {}) or {}
    high_cachegrind = high.get("cachegrind", {}) or {}

    if not isinstance(low_cachegrind, dict):
        raise ValueError("Expected low-priority cachegrind to be a JSON object")
    if not isinstance(high_cachegrind, dict):
        raise ValueError("Expected high-priority cachegrind to be a JSON object")

    if not low_cachegrind and not high_cachegrind:
        return {
            "enabled": False,
            "record_count": 0,
            "planned_record_count": 0,
            "missing_record_count": 0,
            "exclusion_count": 0,
            "records": [],
            "planned_records": [],
            "exclusions": [],
        }

    # High-priority metadata wins, but list-like record containers are merged by
    # stable config/case/params identity so concatenating partial benchmark
    # summaries keeps Cachegrind data from both inputs.
    merged = copy.deepcopy(low_cachegrind)
    merged.update(
        {
            key: copy.deepcopy(value)
            for key, value in high_cachegrind.items()
            if key not in {"records", "planned_records", "exclusions"}
        }
    )

    records_by_key: dict[tuple[str, int, int, int, str, str, str], JsonDict] = {}
    for record in low_cachegrind.get("records", []) or []:
        records_by_key[cachegrind_record_key(record)] = copy.deepcopy(record)
    for record in high_cachegrind.get("records", []) or []:
        records_by_key[cachegrind_record_key(record)] = copy.deepcopy(record)

    planned_by_key: dict[tuple[str, int, int, int, str, str, str], JsonDict] = {}
    for record in low_cachegrind.get("planned_records", []) or []:
        planned_by_key[cachegrind_target_key(record)] = copy.deepcopy(record)
    for record in high_cachegrind.get("planned_records", []) or []:
        planned_by_key[cachegrind_target_key(record)] = copy.deepcopy(record)

    exclusions_by_key: dict[tuple[str, int, int, int, str, str, str], JsonDict] = {}
    for exclusion in low_cachegrind.get("exclusions", []) or []:
        exclusions_by_key[cachegrind_exclusion_key(exclusion)] = copy.deepcopy(exclusion)
    for exclusion in high_cachegrind.get("exclusions", []) or []:
        exclusions_by_key[cachegrind_exclusion_key(exclusion)] = copy.deepcopy(exclusion)

    records = [records_by_key[key] for key in sorted(records_by_key)]
    planned_records = [planned_by_key[key] for key in sorted(planned_by_key)]
    exclusions = [exclusions_by_key[key] for key in sorted(exclusions_by_key)]

    merged["records"] = records
    merged["planned_records"] = planned_records
    merged["exclusions"] = exclusions
    merged["record_count"] = len(records)
    merged["planned_record_count"] = len(planned_records)
    merged["missing_record_count"] = max(0, len(planned_records) - len(records))
    merged["exclusion_count"] = len(exclusions)
    merged["enabled"] = bool(
        merged.get("enabled", False)
        or records
        or planned_records
        or low_cachegrind.get("enabled", False)
        or high_cachegrind.get("enabled", False)
    )

    stats.cachegrind_records = len(records)
    stats.cachegrind_planned_records = len(planned_records)
    stats.cachegrind_exclusions = len(exclusions)
    return merged


def spill_detection_target_key(
    entry: JsonDict,
) -> tuple[str, int, str]:
    covariance_type = entry.get("gmm_covariance_type")
    return (
        str(entry["cpp_case"]),
        _required_int(entry, "D", "spill-detection target"),
        "" if covariance_type is None else str(covariance_type),
    )


def _spill_detection_mapping(
    spill: JsonDict,
    field_name: str,
) -> dict[tuple[str, int, str], JsonDict]:
    entries = spill.get(field_name, []) or []
    if not isinstance(entries, list):
        raise ValueError(
            f"Expected spill_detection.{field_name} to be a list, "
            f"got {type(entries)}"
        )

    indexed: dict[tuple[str, int, str], JsonDict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected entries in spill_detection.{field_name} to be objects, "
                f"got {type(entry)}"
            )
        indexed[spill_detection_target_key(entry)] = copy.deepcopy(entry)
    return indexed


def _spill_detection_errors(spill: JsonDict) -> list[JsonDict]:
    raw_errors = spill.get("errors", []) or []
    if not isinstance(raw_errors, list):
        raise ValueError(
            f"Expected spill_detection.errors to be a list, got {type(raw_errors)}"
        )

    errors: list[JsonDict] = []
    for error in raw_errors:
        if not isinstance(error, dict):
            raise ValueError(
                "Expected entries in spill_detection.errors to be objects, "
                f"got {type(error)}"
            )
        errors.append(copy.deepcopy(error))

    # Raw postprocessing errors predate the list form used by concatenated
    # summaries. Convert the top-level error fields once so repeated merges do
    # not lose them or duplicate an already-normalized error.
    if not errors and (
        spill.get("status") == "ERROR" or spill.get("error") is not None
    ):
        error = {
            key: copy.deepcopy(spill[key])
            for key in (
                "error_type",
                "error",
                "exit_code",
                "out_dir",
                "compile_command_source",
                "scan_target_count",
                "scan_targets",
            )
            if key in spill
        }
        error.setdefault("error_type", "SpillDetectorError")
        error.setdefault("error", "unknown error")
        errors.append(error)

    return errors


def _deduplicate_spill_detection_errors(errors: Iterable[JsonDict]) -> list[JsonDict]:
    deduplicated: dict[str, JsonDict] = {}
    for error in errors:
        signature = json.dumps(error, sort_keys=True, separators=(",", ":"))
        deduplicated[signature] = copy.deepcopy(error)
    return list(deduplicated.values())


def _spill_detection_enabled(spill: JsonDict) -> bool:
    if "enabled" in spill:
        return bool(spill["enabled"])
    return bool(spill)


def _spill_detection_error_summary(errors: list[JsonDict]) -> tuple[str, str]:
    if len(errors) == 1:
        error = errors[0]
        return (
            str(error.get("error_type", "SpillDetectorError")),
            str(error.get("error", "unknown error")),
        )

    descriptions = [
        f"{error.get('error_type', 'SpillDetectorError')}: "
        f"{error.get('error', 'unknown error')}"
        for error in errors
    ]
    return (
        "MultipleSpillDetectorErrors",
        f"{len(errors)} spill-detection errors: " + "; ".join(descriptions),
    )


def merge_spill_detection_summaries(
    low: JsonDict,
    high: JsonDict,
    stats: MergeStats,
) -> JsonDict:
    low_spill = low.get("spill_detection", {}) or {}
    high_spill = high.get("spill_detection", {}) or {}

    if not isinstance(low_spill, dict):
        raise ValueError("Expected low-priority spill_detection to be a JSON object")
    if not isinstance(high_spill, dict):
        raise ValueError("Expected high-priority spill_detection to be a JSON object")

    derived_fields = {
        "enabled",
        "status",
        "scan_target_count",
        "scan_targets",
        "results",
        "total_candidate_reload_pairs",
        "errors",
        "error_count",
        "error_type",
        "error",
        "exit_code",
    }
    merged = {
        key: copy.deepcopy(value)
        for key, value in low_spill.items()
        if key not in derived_fields
    }
    merged.update(
        {
            key: copy.deepcopy(value)
            for key, value in high_spill.items()
            if key not in derived_fields
        }
    )

    # Results imply scan targets, including for legacy summaries that did not
    # store scan_targets explicitly. High-priority duplicates replace their
    # low-priority counterpart atomically.
    targets_by_key = _spill_detection_mapping(low_spill, "scan_targets")
    low_results = _spill_detection_mapping(low_spill, "results")
    for key, result in low_results.items():
        targets_by_key.setdefault(
            key,
            {
                "cpp_case": result["cpp_case"],
                "D": int(result["D"]),
                "gmm_covariance_type": result.get("gmm_covariance_type"),
            },
        )

    targets_by_key.update(_spill_detection_mapping(high_spill, "scan_targets"))
    high_results = _spill_detection_mapping(high_spill, "results")
    for key, result in high_results.items():
        targets_by_key.setdefault(
            key,
            {
                "cpp_case": result["cpp_case"],
                "D": int(result["D"]),
                "gmm_covariance_type": result.get("gmm_covariance_type"),
            },
        )

    results_by_key = low_results
    results_by_key.update(high_results)

    errors = _deduplicate_spill_detection_errors(
        [
            *_spill_detection_errors(low_spill),
            *_spill_detection_errors(high_spill),
        ]
    )
    scan_targets = [targets_by_key[key] for key in sorted(targets_by_key)]
    results = [results_by_key[key] for key in sorted(results_by_key)]
    total_candidate_reload_pairs = sum(
        int(result.get("candidate_reload_pairs", 0)) for result in results
    )
    enabled = bool(
        _spill_detection_enabled(low_spill)
        or _spill_detection_enabled(high_spill)
        or scan_targets
        or results
        or errors
    )

    merged["enabled"] = enabled
    merged["scan_targets"] = scan_targets
    merged["scan_target_count"] = len(scan_targets)
    merged["results"] = results
    merged["errors"] = errors
    merged["error_count"] = len(errors)
    merged["total_candidate_reload_pairs"] = (
        total_candidate_reload_pairs if results or not errors else None
    )

    if errors:
        merged["status"] = "PARTIAL" if results else "ERROR"
        error_type, error_message = _spill_detection_error_summary(errors)
        merged["error_type"] = error_type
        merged["error"] = error_message
        if len(errors) == 1 and "exit_code" in errors[0]:
            merged["exit_code"] = copy.deepcopy(errors[0]["exit_code"])
    elif enabled:
        merged["status"] = "PASS" if total_candidate_reload_pairs == 0 else "FAIL"

    stats.spill_detection_records = len(results)
    stats.spill_detection_scan_targets = len(scan_targets)
    stats.spill_detection_errors = len(errors)
    return merged

def ensure_config_for_exclusion(
    configs: dict[tuple[str, int, int, int], JsonDict], exclusion: JsonDict
) -> JsonDict:
    cfg_key = exclusion_key(exclusion)[:4]
    if cfg_key not in configs:
        configs[cfg_key] = {
            "dataset": cfg_key[0],
            "dimensions": cfg_key[1],
            "samples": cfg_key[2],
            "clusters": cfg_key[3],
            "config_id": exclusion.get("config_id", config_id_for_key(cfg_key)),
            "phases": {},
            "excluded_phases": {},
        }
    return configs[cfg_key]


def rebuild_config_excluded_phases(
    configs: dict[tuple[str, int, int, int], JsonDict], exclusions: list[JsonDict]
) -> None:
    for config in configs.values():
        config["excluded_phases"] = {}

    for exclusion in exclusions:
        config = ensure_config_for_exclusion(configs, exclusion)
        phase_key = str(exclusion.get("phase_key", exclusion.get("phase", "")))
        phase_name = str(exclusion.get("phase", phase_key))
        stage_key = str(exclusion["stage_key"])
        stage_name = str(exclusion["stage"])

        phase_exclusion = config.setdefault("excluded_phases", {}).setdefault(
            phase_name,
            {
                "phase_key": phase_key,
                "phase": phase_name,
                "stages": {},
            },
        )
        phase_exclusion.setdefault("stages", {})[stage_name] = {
            "phase_key": phase_key,
            "phase": phase_name,
            "stage_key": stage_key,
            "stage": stage_name,
            "reason": exclusion.get("reason", ""),
            "matched_rules": copy.deepcopy(exclusion.get("matched_rules", [])),
        }

    for config in configs.values():
        for phase in config.get("excluded_phases", {}).values():
            phase["stages"] = _sorted_nested_mapping(
                phase.get("stages", {}),
                lambda display_key, entry: str(entry["stage_key"]),
            )
        config["excluded_phases"] = _sorted_nested_mapping(
            config.get("excluded_phases", {}),
            lambda display_key, entry: str(entry.get("phase_key", display_key)),
        )


def sorted_configs(configs: dict[tuple[str, int, int, int], JsonDict]) -> list[JsonDict]:
    return [configs[key] for key in sorted(configs)]


def schema_version(summary: JsonDict) -> Any:
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    return metadata.get("schema_version")


def warn_schema_compatibility(low: JsonDict, high: JsonDict, stats: MergeStats) -> None:
    low_schema = schema_version(low)
    high_schema = schema_version(high)
    if low_schema is None or high_schema is None:
        return
    if low_schema != high_schema:
        stats.warnings.append(
            "Merging summaries with different schema versions on a best-effort basis: "
            f"low-priority={low_schema!r}, high-priority={high_schema!r}."
        )


def merge_metadata(
    low: JsonDict,
    high: JsonDict,
    stats: MergeStats,
    *,
    low_path: Path | None = None,
    high_path: Path | None = None,
) -> JsonDict:
    low_metadata = copy.deepcopy(low.get("metadata", {}))
    high_metadata = copy.deepcopy(high.get("metadata", {}))

    if not isinstance(low_metadata, dict):
        low_metadata = {"value": low_metadata}
    if not isinstance(high_metadata, dict):
        high_metadata = {"value": high_metadata}

    # Active metadata follows high priority, but original metadata from both
    # inputs is retained so the merged file remains auditable.
    metadata = copy.deepcopy(low_metadata)
    metadata.update(copy.deepcopy(high_metadata))
    metadata["exclusion_count"] = int(stats.exclusions_kept)
    metadata["merge"] = {
        "policy": (
            "Hierarchical merge by config, phase_key, variant_key, and params_key. "
            "Exact duplicate parameterization blocks are replaced atomically by "
            "the high-priority input. Active exclusions are omitted when concrete "
            "phase data is present in the merged output. Spill-detection targets "
            "and results are merged by cpp_case, D, and gmm_covariance_type."
        ),
        "low_priority_input": str(low_path) if low_path is not None else None,
        "high_priority_input": str(high_path) if high_path is not None else None,
        "counts": {
            "configs_added": stats.configs_added,
            "phases_added": stats.phases_added,
            "stages_added": stats.stages_added,
            "variants_added": stats.variants_added,
            "parameterizations_added": stats.parameterizations_added,
            "parameterizations_replaced": stats.parameterizations_replaced,
            "exclusions_kept": stats.exclusions_kept,
            "exclusions_dropped_due_to_data": stats.exclusions_dropped_due_to_data,
            "compile_artifact_records": stats.compile_artifact_records,
            "cachegrind_records": stats.cachegrind_records,
            "cachegrind_planned_records": stats.cachegrind_planned_records,
            "cachegrind_exclusions": stats.cachegrind_exclusions,
            "spill_detection_records": stats.spill_detection_records,
            "spill_detection_scan_targets": stats.spill_detection_scan_targets,
            "spill_detection_errors": stats.spill_detection_errors,
        },
        "low_priority_metadata": low_metadata,
        "high_priority_metadata": high_metadata,
    }
    return metadata


def merge_benchmark_summaries(
    low: JsonDict,
    high: JsonDict,
    *,
    low_path: Path | None = None,
    high_path: Path | None = None,
) -> tuple[JsonDict, MergeStats]:
    stats = MergeStats()
    warn_schema_compatibility(low, high, stats)

    configs = index_configs(low.get("configs", []), "low-priority configs")
    high_configs = index_configs(high.get("configs", []), "high-priority configs")

    for key, high_config in high_configs.items():
        if key in configs:
            configs[key] = merge_config(configs[key], high_config, stats)
        else:
            configs[key] = copy.deepcopy(high_config)
            configs[key].setdefault("excluded_phases", {})
            stats.configs_added += 1

    exclusions = merge_exclusions(low, high, configs, stats)
    rebuild_config_excluded_phases(configs, exclusions)

    merged = copy.deepcopy(low)
    # Preserve unknown high-priority top-level fields with normal high-priority
    # override semantics, but rebuild the known summary containers explicitly.
    for key, value in high.items():
        if key not in {
            "metadata",
            "configs",
            "exclusions",
            "compile_artifacts",
            "cachegrind",
            "spill_detection",
        }:
            merged[key] = copy.deepcopy(value)

    merged["exclusions"] = exclusions
    merged["compile_artifacts"] = merge_compile_artifacts(low, high, stats)
    merged["cachegrind"] = merge_cachegrind_summaries(low, high, stats)
    if "spill_detection" in low or "spill_detection" in high:
        merged["spill_detection"] = merge_spill_detection_summaries(low, high, stats)
    merged["metadata"] = merge_metadata(
        low,
        high,
        stats,
        low_path=low_path,
        high_path=high_path,
    )
    merged["configs"] = sorted_configs(configs)
    return merged, stats


def main() -> None:
    args = parse_args()
    low = load_json(args.low_priority)
    high = load_json(args.high_priority)
    merged, stats = merge_benchmark_summaries(
        low,
        high,
        low_path=args.low_priority,
        high_path=args.high_priority,
    )
    write_json(args.output, merged)

    for warning in stats.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    print(f"Wrote {args.output}")
    print(f"Configurations: {len(merged.get('configs', []))}")
    print(f"Active exclusions: {len(merged.get('exclusions', []))}")
    print(f"Configs added: {stats.configs_added}")
    print(f"Phases added: {stats.phases_added}")
    print(f"Stages added: {stats.stages_added}")
    print(f"Variants added: {stats.variants_added}")
    print(f"Parameterizations added: {stats.parameterizations_added}")
    print(f"Parameterizations replaced: {stats.parameterizations_replaced}")
    print(f"Exclusions dropped because concrete phase data exists: {stats.exclusions_dropped_due_to_data}")
    print(f"Compile artifact records: {stats.compile_artifact_records}")
    print(f"Cachegrind records: {stats.cachegrind_records}")
    print(f"Cachegrind planned records: {stats.cachegrind_planned_records}")
    print(f"Cachegrind exclusions: {stats.cachegrind_exclusions}")
    print(f"Spill-detection records: {stats.spill_detection_records}")
    print(f"Spill-detection scan targets: {stats.spill_detection_scan_targets}")
    print(f"Spill-detection errors: {stats.spill_detection_errors}")


if __name__ == "__main__":
    main()
