import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_metadata import (
    DEFAULT_DATASET_KEY,
    LANGUAGE_PY_KEY,
    NO_PARAMS,
    REFERENCE_VARIANT,
    format_config_id,
    is_reference_variant,
    language_display_name,
    params_display_name,
    fallback_phase_display_name,
    stage_display_name,
    variant_display_name,
)
MetricsKey = tuple[str, str, str, str, str]
TOKEN_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]*"
VARIANT_PATTERN = rf"(?P<variant>{TOKEN_PATTERN}?)"
STAGE_PATTERN = rf"(?P<stage>{TOKEN_PATTERN}?)"
PARAMS_PATTERN = rf"(?P<params>{TOKEN_PATTERN})"
CONFIG_ID_PATTERN = rf"(?:(?P<dataset>{TOKEN_PATTERN})_)?(?P<D>\d+)D_(?P<N>\d+)N_(?P<K>\d+)K"

# Current timing artifacts:
#   {phase}_{stage}_{variant}_{lang}_{dataset}_{D}D_{N}N_{K}K.json
#   gmm_{stage}_{variant}_{covariance_type}_{lang}_{dataset}_{D}D_{N}N_{K}K.json
# Legacy files without a dataset prefix are parsed as dataset="blobs".
BENCHMARK_JSON_RE = re.compile(
    rf"^(?P<phase>soa|pp|lloyd|hdbscan)_{STAGE_PATTERN}_{VARIANT_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
GMM_BENCHMARK_JSON_RE = re.compile(
    rf"^gmm_{STAGE_PATTERN}_{VARIANT_PATTERN}_{PARAMS_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)

# Current metrics artifacts:
#   lloyd_{stage}_metrics_{variant}_{lang}_{dataset}_{D}D_{N}N_{K}K.json
#   hdbscan_{stage}_metrics_{variant}_{lang}_{dataset}_{D}D_{N}N_{K}K.json
#   gmm_{stage}_metrics_{variant}_{covariance_type}_{lang}_{dataset}_{D}D_{N}N_{K}K.json
LLOYD_METRICS_JSON_RE = re.compile(
    rf"^lloyd_{STAGE_PATTERN}_metrics_{VARIANT_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
GMM_METRICS_JSON_RE = re.compile(
    rf"^gmm_{STAGE_PATTERN}_metrics_{VARIANT_PATTERN}_{PARAMS_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
HDBSCAN_METRICS_JSON_RE = re.compile(
    rf"^hdbscan_{STAGE_PATTERN}_metrics_{VARIANT_PATTERN}_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)


@dataclass(frozen=True, order=True)
class BenchmarkIdentity:
    dimensions: int
    samples: int
    clusters: int
    phase_key: str
    stage_key: str
    variant_key: str
    language_key: str
    params_key: str = NO_PARAMS
    dataset: str = DEFAULT_DATASET_KEY

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "BenchmarkIdentity":
        return cls(
            dimensions=int(record["dimensions"]),
            samples=int(record["samples"]),
            clusters=int(record["clusters"]),
            dataset=str(record.get("dataset", DEFAULT_DATASET_KEY)),
            phase_key=str(record["phase_key"]),
            stage_key=str(record["stage_key"]),
            variant_key=str(record["variant_key"]),
            language_key=str(record["language_key"]),
            params_key=str(record.get("params_key", NO_PARAMS)),
        )

    @property
    def config_id(self) -> str:
        return format_config_id(
            self.dimensions,
            self.samples,
            self.clusters,
            dataset=self.dataset,
        )

    @property
    def config_key(self) -> tuple[str, int, int, int]:
        return (self.dataset, self.dimensions, self.samples, self.clusters)

    @property
    def metrics_key(self) -> MetricsKey:
        return (
            self.config_id,
            self.stage_key,
            self.variant_key,
            self.language_key,
            self.params_key,
        )

    @property
    def phase(self) -> str:
        return fallback_phase_display_name(self.phase_key)

    @property
    def stage(self) -> str:
        return stage_display_name(self.stage_key)

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
            and is_reference_variant(self.phase_key, self.variant_key)
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
            dataset=self.dataset,
            phase_key=self.phase_key,
            stage_key=self.stage_key,
            variant_key=self.variant_key if variant_key is None else variant_key,
            language_key=language_key,
            params_key=self.params_key,
        )

    def python_reference(
        self,
        reference_key: str = REFERENCE_VARIANT,
    ) -> "BenchmarkIdentity":
        return self.with_language(LANGUAGE_PY_KEY, variant_key=reference_key)

    def as_record_fields(self) -> dict[str, Any]:
        return {
            "phase_key": self.phase_key,
            "phase": self.phase,
            "stage_key": self.stage_key,
            "stage": self.stage,
            "variant_key": self.variant_key,
            "variant": self.variant,
            "params_key": self.params_key,
            "params": self.params,
            "language_key": self.language_key,
            "language": self.language,
            "dataset": self.dataset,
            "dimensions": self.dimensions,
            "samples": self.samples,
            "clusters": self.clusters,
            "config_id": self.config_id,
        }


def parse_config_match(match: re.Match[str]) -> tuple[str, int, int, int]:
    dataset = match.group("dataset") or DEFAULT_DATASET_KEY
    D = int(match.group("D"))
    N = int(match.group("N"))
    K = int(match.group("K"))
    return dataset, D, N, K


def _parsed_common(
    match: re.Match[str],
    phase_key: str,
    params_key: str = NO_PARAMS,
) -> BenchmarkIdentity:
    dataset, D, N, K = parse_config_match(match)

    return BenchmarkIdentity(
        dimensions=D,
        samples=N,
        clusters=K,
        dataset=dataset,
        phase_key=phase_key,
        stage_key=match.group("stage"),
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
    elif phase_key == "hdbscan":
        match = HDBSCAN_METRICS_JSON_RE.match(path.name)
        params_key = NO_PARAMS
    else:
        raise ValueError(f"No metrics filename parser for phase {phase_key!r}")

    if not match:
        return None

    return _parsed_common(
        match,
        phase_key=phase_key,
        params_key=params_key,
    )
