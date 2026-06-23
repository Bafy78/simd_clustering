import json
import shutil
import sys
from pathlib import Path
from typing import Any

from benchmark_pipeline.cpp_cases import get_cpp_case
from benchmark_metadata import stage_display_name
from benchmark_pipeline.exclusions import CachegrindExclusionRule, phase_display_name
from benchmark_pipeline.paths import REPO_ROOT, repo_relative_path

CACHEGRIND_SCHEMA_VERSION = 1
CACHEGRIND_MANIFEST_SCHEMA_VERSION = 1
CACHEGRIND_RESULTS_DIR = Path("callgrind_results")
CACHEGRIND_PREFIX = "cachegrind"
CACHEGRIND_MANIFEST_FILENAME = "cachegrind_manifest.json"
PROFILE_EVENTS = (
    "Ir",
    "I1mr",
    "ILmr",
    "Dr",
    "D1mr",
    "DLmr",
    "Dw",
    "D1mw",
    "DLmw",
)


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"Required tool not found on PATH: {tool}")
        sys.exit(1)


def require_cachegrind_tools() -> None:
    require_tool("valgrind")
    require_tool("callgrind_annotate")


def profile_events_string() -> str:
    return ",".join(PROFILE_EVENTS)


def cachegrind_file_stem(
    cpp_case: str,
    stage_key: str,
    params_key: str,
    case_id: str,
) -> str:
    return f"cachegrind.{cpp_case}.{stage_key}.{params_key}.{case_id}"


def cachegrind_summary_filename(
    cpp_case: str,
    stage_key: str,
    params_key: str,
    case_id: str,
) -> str:
    return f"{CACHEGRIND_PREFIX}.{cpp_case}.{stage_key}.{params_key}.{case_id}.json"


def cachegrind_summary_path(
    out_dir: str | Path,
    cpp_case: str,
    stage_key: str,
    params_key: str,
    case_id: str,
) -> Path:
    return (
        repo_relative_path(out_dir)
        / cachegrind_summary_filename(cpp_case, stage_key, params_key, case_id)
    )


def cachegrind_manifest_path(out_dir: str | Path) -> Path:
    return repo_relative_path(out_dir) / CACHEGRIND_MANIFEST_FILENAME


def prepare_cachegrind_results_dir(out_dir: str | Path) -> Path:
    """Clean and recreate the Cachegrind result directory for a fresh pipeline run."""
    out_dir = repo_relative_path(out_dir)
    protected = {REPO_ROOT.resolve(), REPO_ROOT.parent.resolve(), Path(out_dir.anchor).resolve()}

    if out_dir.resolve() in protected:
        raise ValueError(f"Refusing to clean protected Cachegrind output directory: {out_dir}")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"Cachegrind output path exists but is not a directory: {out_dir}")
    if out_dir.is_symlink():
        raise ValueError(f"Refusing to clean symlinked Cachegrind output directory: {out_dir}")

    if out_dir.exists():
        print(f"Cleaning {out_dir}...")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def add_cache_options(
    command: list[str],
    *,
    I1: str | None = None,
    D1: str | None = None,
    LL: str | None = None,
) -> None:
    if I1:
        command.append(f"--I1={I1}")
    if D1:
        command.append(f"--D1={D1}")
    if LL:
        command.append(f"--LL={LL}")


def cache_model_record(
    *,
    I1: str | None = None,
    D1: str | None = None,
    LL: str | None = None,
) -> dict[str, str | None]:
    return {
        "I1": I1,
        "D1": D1,
        "LL": LL,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def derived_cache_metrics(events: dict[str, int]) -> dict[str, int | float | None]:
    instruction_refs = int(events.get("Ir", 0))
    data_reads = int(events.get("Dr", 0))
    data_writes = int(events.get("Dw", 0))
    data_refs = data_reads + data_writes

    d1_read_misses = int(events.get("D1mr", 0))
    d1_write_misses = int(events.get("D1mw", 0))
    ll_read_misses = int(events.get("DLmr", 0))
    ll_write_misses = int(events.get("DLmw", 0))
    d1_data_misses = d1_read_misses + d1_write_misses
    ll_data_misses = ll_read_misses + ll_write_misses

    return {
        "instruction_refs": instruction_refs,
        "instruction_l1_misses": int(events.get("I1mr", 0)),
        "instruction_ll_misses": int(events.get("ILmr", 0)),
        "instruction_l1_miss_rate": _safe_rate(
            int(events.get("I1mr", 0)), instruction_refs
        ),
        "instruction_ll_miss_rate": _safe_rate(
            int(events.get("ILmr", 0)), instruction_refs
        ),
        "data_reads": data_reads,
        "data_writes": data_writes,
        "data_refs": data_refs,
        "d1_read_misses": d1_read_misses,
        "d1_write_misses": d1_write_misses,
        "d1_data_misses": d1_data_misses,
        "ll_read_misses": ll_read_misses,
        "ll_write_misses": ll_write_misses,
        "ll_data_misses": ll_data_misses,
        "d1_data_miss_rate": _safe_rate(d1_data_misses, data_refs),
        "ll_data_miss_rate": _safe_rate(ll_data_misses, data_refs),
        "d1_read_miss_rate": _safe_rate(d1_read_misses, data_reads),
        "d1_write_miss_rate": _safe_rate(d1_write_misses, data_writes),
        "ll_read_miss_rate": _safe_rate(ll_read_misses, data_reads),
        "ll_write_miss_rate": _safe_rate(ll_write_misses, data_writes),
    }


def parse_cachegrind_summary_events(path: str | Path) -> dict[str, int]:
    path = Path(path)
    events: list[str] | None = None
    summary_values: list[int] | None = None
    totals_values: list[int] | None = None

    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith("events:"):
                events = line.split(":", 1)[1].strip().split()

            elif line.startswith("summary:"):
                values_text = line.split(":", 1)[1].strip().split()
                summary_values = [int(value, 0) for value in values_text]

            elif line.startswith("totals:"):
                values_text = line.split(":", 1)[1].strip().split()
                totals_values = [int(value, 0) for value in values_text]

    if events is None:
        raise ValueError(f"No events line found in Cachegrind output: {path}")

    values = totals_values if totals_values is not None else summary_values
    values_label = "totals" if totals_values is not None else "summary"

    if values is None:
        raise ValueError(f"No totals or summary line found in Cachegrind output: {path}")

    if len(events) != len(values):
        raise ValueError(
            f"Cachegrind events/{values_label} length mismatch in {path}: "
            f"{len(events)} events vs {len(values)} values"
        )

    result = {event: value for event, value in zip(events, values)}

    missing = [event for event in PROFILE_EVENTS if event not in result]
    if missing:
        raise ValueError(
            f"Cachegrind output {path} is missing expected cache-sim events: {missing}"
        )

    return {event: int(result[event]) for event in PROFILE_EVENTS}


def _repo_relative_or_string(path: str | Path | None) -> str | None:
    if path is None:
        return None

    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_cachegrind_record(
    *,
    cpp_case: str,
    D: int,
    N: int,
    K: int,
    params_key: str,
    cache_model: dict[str, str | None],
    events: dict[str, int],
    raw_output: str | Path,
    annotated_output: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
    annotate_stderr_path: str | Path,
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    case = get_cpp_case(cpp_case)
    files: dict[str, str | None] = {
        "raw": _repo_relative_or_string(raw_output),
        "annotated": _repo_relative_or_string(annotated_output),
        "stdout": _repo_relative_or_string(stdout_path),
        "stderr": _repo_relative_or_string(stderr_path),
        "annotate_stderr": _repo_relative_or_string(annotate_stderr_path),
    }
    if metrics_path is not None:
        files["metrics"] = _repo_relative_or_string(metrics_path)

    return {
        "schema_version": CACHEGRIND_SCHEMA_VERSION,
        "tool": "callgrind",
        "cache_sim": True,
        "cache_sim_source": "Cachegrind counters",
        "cpp_case": cpp_case,
        "phase_key": case.phase_key,
        "phase": phase_display_name(case.phase_key),
        "stage_key": case.stage_key,
        "stage": stage_display_name(case.stage_key),
        "variant_key": case.variant_key,
        "variant": case.variant_key.replace("_", " ").title(),
        "params_key": params_key,
        "D": int(D),
        "N": int(N),
        "K": int(K),
        "config_id": f"{int(D)}D_{int(N)}N_{int(K)}K",
        "cache_model": cache_model,
        "events": {event: int(events[event]) for event in PROFILE_EVENTS},
        "derived": derived_cache_metrics(events),
        "files": files,
    }


def build_cachegrind_exclusion_record(
    *,
    D: int,
    N: int,
    K: int,
    cpp_case: str,
    params_key: str,
    rules: tuple[CachegrindExclusionRule, ...],
) -> dict[str, Any] | None:
    case = get_cpp_case(cpp_case)
    matched_rules = [
        {
            "rule_index": rule_index,
            "reason": rule.reason,
        }
        for rule_index, rule in enumerate(rules)
        if rule.matches(
            D=D,
            N=N,
            K=K,
            phase_key=case.phase_key,
            stage_key=case.stage_key,
            cpp_case=cpp_case,
            params_key=params_key,
        )
    ]

    if not matched_rules:
        return None

    reasons = list(dict.fromkeys(record["reason"] for record in matched_rules))
    return {
        "config_id": f"{D}D_{N}N_{K}K",
        "dimensions": int(D),
        "samples": int(N),
        "clusters": int(K),
        "cpp_case": cpp_case,
        "phase_key": case.phase_key,
        "phase": phase_display_name(case.phase_key),
        "stage_key": case.stage_key,
        "stage": stage_display_name(case.stage_key),
        "variant_key": case.variant_key,
        "params_key": params_key,
        "reason": " ; ".join(reasons),
        "matched_rules": matched_rules,
    }


def build_cachegrind_manifest(
    *,
    enabled: bool,
    results_dir: str | Path,
    cache_model: dict[str, str | None],
    planned_records: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CACHEGRIND_MANIFEST_SCHEMA_VERSION,
        "description": (
            "Cachegrind manifest. Planned records are C++ "
            "config/case/parameterization targets not excluded by Cachegrind-specific "
            "rules. Exclusions skip only Cachegrind, not normal timing."
        ),
        "enabled": bool(enabled),
        "results_dir": _repo_relative_or_string(results_dir),
        "cache_model": cache_model,
        "planned_record_count": len(planned_records),
        "exclusion_count": len(exclusions),
        "planned_records": sorted(
            planned_records,
            key=lambda item: (
                int(item["D"]),
                int(item["N"]),
                int(item["K"]),
                str(item["stage_key"]),
                str(item["cpp_case"]),
                str(item["params_key"]),
            ),
        ),
        "exclusions": sorted(
            exclusions,
            key=lambda item: (
                int(item["dimensions"]),
                int(item["samples"]),
                int(item["clusters"]),
                str(item["cpp_case"]),
                str(item["params_key"]),
            ),
        ),
    }
