import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from benchmark_metadata import (
    DEFAULT_DATASET_KEY,
    FULL_STAGE_KEY,
    NO_PARAMS,
    PHASE_KEYS,
    all_stage_keys,
    fallback_phase_display_name,
    stage_display_name,
)

VALID_PHASE_KEYS = PHASE_KEYS
VALID_STAGE_KEYS = all_stage_keys()
EXCLUSIONS_FILENAME = "benchmark_exclusions.json"

IntSelector = int | tuple[int, ...] | list[int] | set[int] | None
PhaseSelector = str | tuple[str, ...] | list[str] | set[str] | None
StageSelector = str | tuple[str, ...] | list[str] | set[str] | None
StringSelector = str | tuple[str, ...] | list[str] | set[str] | None


@dataclass(frozen=True)
class BenchmarkExclusionRule:
    """
    Exclude benchmark work for selected D/N/K combinations.

    Rules are phase- and stage-aware so one D/N/K point can be skipped for a
    specific phase stage while other stages still run. Leave phase_keys or
    stage_keys as None to apply to every enabled phase or stage. Exact selectors
    (dimensions/samples/clusters) can be combined with min_/max_ bounds.
    """

    reason: str
    phase_keys: PhaseSelector = None
    stage_keys: StageSelector = None
    datasets: StringSelector = None
    dimensions: IntSelector = None
    samples: IntSelector = None
    clusters: IntSelector = None
    min_dimensions: int | None = None
    max_dimensions: int | None = None
    min_samples: int | None = None
    max_samples: int | None = None
    min_clusters: int | None = None
    max_clusters: int | None = None

    def resolved_phase_keys(self) -> tuple[str, ...]:
        phase_keys = _normalize_phase_selector(self.phase_keys)
        unknown = sorted(set(phase_keys) - set(VALID_PHASE_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown exclusion phase key(s) {unknown}. "
                f"Expected one or more of: {', '.join(VALID_PHASE_KEYS)}"
            )
        return phase_keys

    def resolved_stage_keys(self) -> tuple[str, ...] | None:
        stage_keys = _normalize_stage_selector(self.stage_keys)
        if stage_keys is None:
            return None
        unknown = sorted(set(stage_keys) - set(VALID_STAGE_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown exclusion stage key(s) {unknown}. "
                f"Expected one or more of: {', '.join(VALID_STAGE_KEYS)}"
            )
        return stage_keys

    def resolved_datasets(self) -> tuple[str, ...] | None:
        return _normalize_string_selector(self.datasets)

    def matches(
        self,
        *,
        dataset: str = DEFAULT_DATASET_KEY,
        D: int,
        N: int,
        K: int,
        phase_key: str,
        stage_key: str = FULL_STAGE_KEY,
    ) -> bool:
        if not self.reason.strip():
            raise ValueError("Benchmark exclusion rules require a non-empty reason.")

        if phase_key not in self.resolved_phase_keys():
            return False
        resolved_stage_keys = self.resolved_stage_keys()
        if resolved_stage_keys is not None and stage_key not in resolved_stage_keys:
            return False

        resolved_datasets = self.resolved_datasets()
        if resolved_datasets is not None and dataset not in resolved_datasets:
            return False

        return (
            _value_matches(D, self.dimensions, self.min_dimensions, self.max_dimensions)
            and _value_matches(N, self.samples, self.min_samples, self.max_samples)
            and _value_matches(K, self.clusters, self.min_clusters, self.max_clusters)
        )


@dataclass(frozen=True)
class CachegrindExclusionRule:
    """
    Exclude Cachegrind work for selected D/N/K/C++ case targets.

    These rules are intentionally separate from BenchmarkExclusionRule: they skip
    only the Valgrind/Callgrind-based Cachegrind pass and leave the normal timing suite
    unchanged. Leave cpp_cases, phase_keys, stage_keys, or params_keys as None to match all.
    """

    reason: str
    cpp_cases: StringSelector = None
    phase_keys: PhaseSelector = None
    stage_keys: StageSelector = None
    params_keys: StringSelector = None
    datasets: StringSelector = None
    dimensions: IntSelector = None
    samples: IntSelector = None
    clusters: IntSelector = None
    min_dimensions: int | None = None
    max_dimensions: int | None = None
    min_samples: int | None = None
    max_samples: int | None = None
    min_clusters: int | None = None
    max_clusters: int | None = None

    def resolved_phase_keys(self) -> tuple[str, ...]:
        phase_keys = _normalize_phase_selector(self.phase_keys)
        unknown = sorted(set(phase_keys) - set(VALID_PHASE_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown Cachegrind exclusion phase key(s) {unknown}. "
                f"Expected one or more of: {', '.join(VALID_PHASE_KEYS)}"
            )
        return phase_keys

    def resolved_stage_keys(self) -> tuple[str, ...] | None:
        stage_keys = _normalize_stage_selector(self.stage_keys)
        if stage_keys is None:
            return None
        unknown = sorted(set(stage_keys) - set(VALID_STAGE_KEYS))
        if unknown:
            raise ValueError(
                f"Unknown Cachegrind exclusion stage key(s) {unknown}. "
                f"Expected one or more of: {', '.join(VALID_STAGE_KEYS)}"
            )
        return stage_keys

    def resolved_cpp_cases(self) -> tuple[str, ...] | None:
        return _normalize_string_selector(self.cpp_cases)

    def resolved_params_keys(self) -> tuple[str, ...] | None:
        return _normalize_string_selector(self.params_keys)

    def resolved_datasets(self) -> tuple[str, ...] | None:
        return _normalize_string_selector(self.datasets)

    def matches(
        self,
        *,
        dataset: str = DEFAULT_DATASET_KEY,
        D: int,
        N: int,
        K: int,
        phase_key: str,
        stage_key: str = FULL_STAGE_KEY,
        cpp_case: str,
        params_key: str = NO_PARAMS,
    ) -> bool:
        if not self.reason.strip():
            raise ValueError("Cachegrind exclusion rules require a non-empty reason.")

        if phase_key not in self.resolved_phase_keys():
            return False
        resolved_stage_keys = self.resolved_stage_keys()
        if resolved_stage_keys is not None and stage_key not in resolved_stage_keys:
            return False

        cpp_cases = self.resolved_cpp_cases()
        if cpp_cases is not None and cpp_case not in cpp_cases:
            return False

        params_keys = self.resolved_params_keys()
        if params_keys is not None and params_key not in params_keys:
            return False

        resolved_datasets = self.resolved_datasets()
        if resolved_datasets is not None and dataset not in resolved_datasets:
            return False

        return (
            _value_matches(D, self.dimensions, self.min_dimensions, self.max_dimensions)
            and _value_matches(N, self.samples, self.min_samples, self.max_samples)
            and _value_matches(K, self.clusters, self.min_clusters, self.max_clusters)
        )


def _normalize_phase_selector(value: PhaseSelector) -> tuple[str, ...]:
    if value is None:
        return VALID_PHASE_KEYS
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_stage_selector(value: StageSelector) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_string_selector(value: StringSelector) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _normalize_int_selector(value: IntSelector) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value,)
    return tuple(int(item) for item in value)


def _value_matches(
    value: int,
    exact_selector: IntSelector,
    min_value: int | None,
    max_value: int | None,
) -> bool:
    exact_values = _normalize_int_selector(exact_selector)
    if exact_values is not None and value not in exact_values:
        return False
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def phase_display_name(phase_key: str) -> str:
    return fallback_phase_display_name(phase_key)


def exclusion_records_for_case(
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
    stage_keys_by_phase: dict[str, Iterable[str]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rules = tuple(rules)

    for phase_key in phase_keys:
        stage_keys = tuple(
            stage_keys_by_phase.get(phase_key, (FULL_STAGE_KEY,))
            if stage_keys_by_phase is not None
            else (FULL_STAGE_KEY,)
        )
        for stage_key in stage_keys:
            matched_rules = [
                {
                    "rule_index": rule_index,
                    "reason": rule.reason,
                }
                for rule_index, rule in enumerate(rules)
                if rule.matches(
                    dataset=dataset,
                    D=D,
                    N=N,
                    K=K,
                    phase_key=phase_key,
                    stage_key=stage_key,
                )
            ]

            if not matched_rules:
                continue

            reasons = list(dict.fromkeys(rule_record["reason"] for rule_record in matched_rules))
            records.append(
                {
                    "config_id": f"{dataset}_{D}D_{N}N_{K}K",
                    "dataset": dataset,
                    "dimensions": int(D),
                    "samples": int(N),
                    "clusters": int(K),
                    "phase_key": phase_key,
                    "phase": phase_display_name(phase_key),
                    "stage_key": stage_key,
                    "stage": stage_display_name(stage_key),
                    "reason": " ; ".join(reasons),
                    "matched_rules": matched_rules,
                }
            )

    return records


def excluded_phase_keys_for_case(
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
    stage_keys_by_phase: dict[str, Iterable[str]] | None = None,
) -> set[str]:
    """Return phases whose enabled stages are all excluded for this case."""

    normalized_stage_keys = {
        phase_key: tuple(
            stage_keys_by_phase.get(phase_key, (FULL_STAGE_KEY,))
            if stage_keys_by_phase is not None
            else (FULL_STAGE_KEY,)
        )
        for phase_key in phase_keys
    }
    excluded_stage_keys: dict[str, set[str]] = {}
    for record in exclusion_records_for_case(
        dataset=dataset,
        D=D,
        N=N,
        K=K,
        rules=rules,
        phase_keys=phase_keys,
        stage_keys_by_phase=normalized_stage_keys,
    ):
        excluded_stage_keys.setdefault(record["phase_key"], set()).add(
            record["stage_key"]
        )

    return {
        phase_key
        for phase_key, stage_keys in normalized_stage_keys.items()
        if stage_keys and set(stage_keys).issubset(excluded_stage_keys.get(phase_key, set()))
    }


def excluded_phase_stage_keys_for_case(
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
    stage_keys_by_phase: dict[str, Iterable[str]] | None = None,
) -> set[tuple[str, str]]:
    return {
        (record["phase_key"], record["stage_key"])
        for record in exclusion_records_for_case(
            dataset=dataset,
            D=D,
            N=N,
            K=K,
            rules=rules,
            phase_keys=phase_keys,
            stage_keys_by_phase=stage_keys_by_phase,
        )
    }


def is_phase_excluded(
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
    phase_key: str,
    stage_key: str = FULL_STAGE_KEY,
    rules: Iterable[BenchmarkExclusionRule],
) -> bool:
    return bool(
        exclusion_records_for_case(
            dataset=dataset,
            D=D,
            N=N,
            K=K,
            rules=rules,
            phase_keys=(phase_key,),
            stage_keys_by_phase={phase_key: (stage_key,)},
        )
    )


def is_cachegrind_excluded(
    *,
    dataset: str = DEFAULT_DATASET_KEY,
    D: int,
    N: int,
    K: int,
    phase_key: str,
    stage_key: str = FULL_STAGE_KEY,
    cpp_case: str,
    params_key: str = NO_PARAMS,
    rules: Iterable[CachegrindExclusionRule],
) -> bool:
    return any(
        rule.matches(
            dataset=dataset,
            D=D,
            N=N,
            K=K,
            phase_key=phase_key,
            stage_key=stage_key,
            cpp_case=cpp_case,
            params_key=params_key,
        )
        for rule in rules
    )


def _manifest_case_tuple(case: Any) -> tuple[str, int, int, int]:
    if hasattr(case, "dataset") and hasattr(case, "D") and hasattr(case, "N") and hasattr(case, "K"):
        return (str(case.dataset), int(case.D), int(case.N), int(case.K))

    if isinstance(case, dict):
        return (
            str(case.get("dataset", DEFAULT_DATASET_KEY)),
            int(case["D"]),
            int(case["N"]),
            int(case["K"]),
        )

    dataset, D, N, K = case
    return str(dataset), int(D), int(N), int(K)


def build_exclusion_manifest(
    *,
    test_Ds: Iterable[int] | None = None,
    test_Ns: Iterable[int] | None = None,
    test_Ks: Iterable[int] | None = None,
    cases: Iterable[Any] | None = None,
    dataset: str = DEFAULT_DATASET_KEY,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
    stage_keys_by_phase: dict[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    rules = tuple(rules)
    exclusions: list[dict[str, Any]] = []

    if cases is None:
        if test_Ds is None or test_Ns is None or test_Ks is None:
            raise ValueError("Either cases or test_Ds/test_Ns/test_Ks must be provided.")
        case_tuples = (
            (dataset, int(D), int(N), int(K))
            for D in test_Ds
            for N in test_Ns
            for K in test_Ks
        )
    else:
        case_tuples = (_manifest_case_tuple(case) for case in cases)

    for dataset_key, D, N, K in case_tuples:
        exclusions.extend(
            exclusion_records_for_case(
                dataset=dataset_key,
                D=D,
                N=N,
                K=K,
                rules=rules,
                phase_keys=phase_keys,
                stage_keys_by_phase=stage_keys_by_phase,
            )
        )

    exclusions.sort(
        key=lambda item: (
            str(item.get("dataset", DEFAULT_DATASET_KEY)),
            item["dimensions"],
            item["samples"],
            item["clusters"],
            item["phase_key"],
            item["stage_key"],
        )
    )

    return {
        "schema_version": 3,
        "description": (
            "User-configured benchmark exclusions. Each entry is a "
            "dataset/D/N/K/phase/stage combination that the orchestrator intentionally skipped."
        ),
        "exclusions": exclusions,
    }

def write_exclusion_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def load_exclusion_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "schema_version": 2,
            "description": "No benchmark exclusion manifest was found.",
            "exclusions": [],
        }

    with path.open("r") as f:
        manifest = json.load(f)

    if "exclusions" not in manifest:
        raise ValueError(f"Invalid benchmark exclusion manifest: {path}")

    for exclusion in manifest.get("exclusions", []):
        if "stage_key" not in exclusion or "stage" not in exclusion:
            raise ValueError(
                f"Invalid benchmark exclusion manifest entry without stage: {exclusion!r}"
            )

    return manifest
