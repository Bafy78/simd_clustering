import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

VALID_PHASE_KEYS = ("soa", "pp", "lloyd", "gmm")
PHASE_DISPLAY_NAMES = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Algorithm",
    "gmm": "GaussianMixture EM",
}
EXCLUSIONS_FILENAME = "benchmark_exclusions.json"

IntSelector = int | tuple[int, ...] | list[int] | set[int] | None
PhaseSelector = str | tuple[str, ...] | list[str] | set[str] | None


@dataclass(frozen=True)
class BenchmarkExclusionRule:
    """
    Exclude benchmark work for selected D/N/K combinations.

    Rules are phase-aware so one D/N/K point can be skipped for GMM while Lloyd,
    K-Means++, or the SoA tax still runs. Leave phase_keys as None to apply to
    every phase. Exact selectors (dimensions/samples/clusters) can be combined
    with min_/max_ bounds.
    """

    reason: str
    phase_keys: PhaseSelector = None
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

    def matches(self, *, D: int, N: int, K: int, phase_key: str) -> bool:
        if not self.reason.strip():
            raise ValueError("Benchmark exclusion rules require a non-empty reason.")

        if phase_key not in self.resolved_phase_keys():
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
    return PHASE_DISPLAY_NAMES.get(phase_key, phase_key)


def exclusion_records_for_case(
    *,
    D: int,
    N: int,
    K: int,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    rules = tuple(rules)

    for phase_key in phase_keys:
        matched_rules = [
            {
                "rule_index": rule_index,
                "reason": rule.reason,
            }
            for rule_index, rule in enumerate(rules)
            if rule.matches(D=D, N=N, K=K, phase_key=phase_key)
        ]

        if not matched_rules:
            continue

        # Keep a compact human-facing reason while preserving individual rules.
        reasons = list(dict.fromkeys(rule_record["reason"] for rule_record in matched_rules))
        records.append(
            {
                "config_id": f"{D}D_{N}N_{K}K",
                "dimensions": int(D),
                "samples": int(N),
                "clusters": int(K),
                "phase_key": phase_key,
                "phase": phase_display_name(phase_key),
                "reason": " ; ".join(reasons),
                "matched_rules": matched_rules,
            }
        )

    return records


def excluded_phase_keys_for_case(
    *,
    D: int,
    N: int,
    K: int,
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
) -> set[str]:
    return {
        record["phase_key"]
        for record in exclusion_records_for_case(
            D=D,
            N=N,
            K=K,
            rules=rules,
            phase_keys=phase_keys,
        )
    }


def is_phase_excluded(
    *,
    D: int,
    N: int,
    K: int,
    phase_key: str,
    rules: Iterable[BenchmarkExclusionRule],
) -> bool:
    return bool(
        exclusion_records_for_case(
            D=D,
            N=N,
            K=K,
            rules=rules,
            phase_keys=(phase_key,),
        )
    )


def build_exclusion_manifest(
    *,
    test_Ds: Iterable[int],
    test_Ns: Iterable[int],
    test_Ks: Iterable[int],
    rules: Iterable[BenchmarkExclusionRule],
    phase_keys: Iterable[str] = VALID_PHASE_KEYS,
) -> dict[str, Any]:
    rules = tuple(rules)
    exclusions: list[dict[str, Any]] = []

    for D in test_Ds:
        for N in test_Ns:
            for K in test_Ks:
                exclusions.extend(
                    exclusion_records_for_case(
                        D=int(D),
                        N=int(N),
                        K=int(K),
                        rules=rules,
                        phase_keys=phase_keys,
                    )
                )

    exclusions.sort(
        key=lambda item: (
            item["dimensions"],
            item["samples"],
            item["clusters"],
            item["phase_key"],
        )
    )

    return {
        "schema_version": 1,
        "description": (
            "User-configured benchmark exclusions. Each entry is a D/N/K/phase "
            "combination that the orchestrator intentionally skipped."
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
            "schema_version": 1,
            "description": "No benchmark exclusion manifest was found.",
            "exclusions": [],
        }

    with path.open("r") as f:
        manifest = json.load(f)

    if "exclusions" not in manifest:
        raise ValueError(f"Invalid benchmark exclusion manifest: {path}")

    return manifest
