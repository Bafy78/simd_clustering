import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_metadata import (
    LANGUAGE_PY_KEY,
    NO_PARAMS,
    REFERENCE_VARIANT,
    format_config_id,
    language_display_name,
    params_display_name,
    phase_display_name,
    variant_display_name,
)
MetricsKey = tuple[str, str, str, str]
TOKEN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]*"
VARIANT_PATTERN = rf"(?P<variant>{TOKEN_PATTERN})"
PARAMS_PATTERN = rf"(?P<params>{TOKEN_PATTERN})"
CONFIG_ID_PATTERN = r"(?P<D>\d+)D_(?P<N>\d+)N_(?P<K>\d+)K"

# Timing artifacts:
#   {phase}_{variant}_{lang}_{D}D_{N}N_{K}K.json
#   gmm_{variant}_{covariance_type}_{lang}_{D}D_{N}N_{K}K.json
BENCHMARK_JSON_RE = re.compile(
    rf"^(?P<phase>soa|pp|lloyd)_{VARIANT_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
GMM_BENCHMARK_JSON_RE = re.compile(
    rf"^gmm_{VARIANT_PATTERN}_{PARAMS_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)

# Metrics artifacts:
#   lloyd_metrics_{variant}_{lang}_{D}D_{N}N_{K}K.json
#   gmm_metrics_{variant}_{covariance_type}_{lang}_{D}D_{N}N_{K}K.json
LLOYD_METRICS_JSON_RE = re.compile(
    rf"^lloyd_metrics_{VARIANT_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
GMM_METRICS_JSON_RE = re.compile(
    rf"^gmm_metrics_{VARIANT_PATTERN}_{PARAMS_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)


@dataclass(frozen=True, order=True)
class BenchmarkIdentity:
    dimensions: int
    samples: int
    clusters: int
    phase_key: str
    variant_key: str
    language_key: str
    params_key: str = NO_PARAMS

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BenchmarkIdentity":
        return cls(
            dimensions=int(record["dimensions"]),
            samples=int(record["samples"]),
            clusters=int(record["clusters"]),
            phase_key=str(record["phase_key"]),
            variant_key=str(record["variant_key"]),
            language_key=str(record["language_key"]),
            params_key=str(record.get("params_key", NO_PARAMS)),
        )

    @property
    def config_id(self) -> str:
        return format_config_id(self.dimensions, self.samples, self.clusters)

    @property
    def config_key(self) -> tuple[int, int, int]:
        return (self.dimensions, self.samples, self.clusters)

    @property
    def metrics_key(self) -> MetricsKey:
        return (
            self.config_id,
            self.variant_key,
            self.language_key,
            self.params_key,
        )

    @property
    def phase(self) -> str:
        return phase_display_name(self.phase_key)

    @property
    def variant(self) -> str:
        return variant_display_name(self.variant_key)

    @property
    def params(self) -> str:
        return params_display_name(self.params_key)

    @property
    def language(self) -> str:
        return language_display_name(self.language_key)

    @property
    def is_python_reference(self) -> bool:
        return (
            self.language_key == LANGUAGE_PY_KEY
            and self.variant_key == REFERENCE_VARIANT
        )

    def with_language(
        self,
        language_key: str,
        variant_key: str | None = None,
    ) -> "BenchmarkIdentity":
        return BenchmarkIdentity(
            dimensions=self.dimensions,
            samples=self.samples,
            clusters=self.clusters,
            phase_key=self.phase_key,
            variant_key=self.variant_key if variant_key is None else variant_key,
            language_key=language_key,
            params_key=self.params_key,
        )

    def python_reference(self) -> "BenchmarkIdentity":
        return self.with_language(LANGUAGE_PY_KEY, variant_key=REFERENCE_VARIANT)

    def as_record_fields(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "phase": self.phase,
            "variant_key": self.variant_key,
            "variant": self.variant,
            "params_key": self.params_key,
            "params": self.params,
            "language_key": self.language_key,
            "language": self.language,
            "dimensions": self.dimensions,
            "samples": self.samples,
            "clusters": self.clusters,
            "config_id": self.config_id,
        }


def parse_config_match(match: re.Match[str]) -> tuple[int, int, int]:
    D = int(match.group("D"))
    N = int(match.group("N"))
    K = int(match.group("K"))
    return D, N, K


def _parsed_common(
    match: re.Match[str],
    phase_key: str,
    params_key: str = NO_PARAMS,
) -> BenchmarkIdentity:
    D, N, K = parse_config_match(match)

    return BenchmarkIdentity(
        dimensions=D,
        samples=N,
        clusters=K,
        phase_key=phase_key,
        variant_key=match.group("variant"),
        language_key=match.group("lang"),
        params_key=params_key,
    )


def parse_benchmark_filename(path: Path) -> BenchmarkIdentity | None:
    match = BENCHMARK_JSON_RE.match(path.name)
    if match:
        return _parsed_common(match, phase_key=match.group("phase"))

    match = GMM_BENCHMARK_JSON_RE.match(path.name)
    if match:
        return _parsed_common(
            match,
            phase_key="gmm",
            params_key=match.group("params"),
        )

    return None


def parse_metrics_filename(path: Path, phase_key: str) -> BenchmarkIdentity | None:
    if phase_key == "lloyd":
        match = LLOYD_METRICS_JSON_RE.match(path.name)
        params_key = NO_PARAMS
    elif phase_key == "gmm":
        match = GMM_METRICS_JSON_RE.match(path.name)
        params_key = match.group("params") if match else NO_PARAMS
    else:
        raise ValueError(f"No metrics filename parser for phase {phase_key!r}")

    if not match:
        return None

    return _parsed_common(match, phase_key=phase_key, params_key=params_key)
