import re
from pathlib import Path
from typing import Any

PHASE_MAP = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Algorithm",
    "gmm": "GaussianMixture EM",
}

LANG_MAP = {
    "cpp": "C++",
    "py": "Python",
}

VARIANT_MAP = {
    "static": "Static",
    "dynamic": "Dynamic",
    "auto": "Auto",
    "reference": "Reference",
}

NO_PARAMS = "default"
LANGUAGE_REFERENCE_VARIANT = "reference"
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


def format_config_id(D: int, N: int, K: int) -> str:
    return f"{D}D_{N}N_{K}K"


def parse_config_match(match: re.Match[str]) -> tuple[int, int, int]:
    D = int(match.group("D"))
    N = int(match.group("N"))
    K = int(match.group("K"))
    return D, N, K


def display_name(key: str) -> str:
    return key.replace("_", " ").title()


def variant_display_name(variant_key: str) -> str:
    return VARIANT_MAP.get(variant_key, display_name(variant_key))


def params_display_name(params_key: str) -> str:
    if params_key == NO_PARAMS:
        return "Default"
    return display_name(params_key)


def _parsed_common(
    match: re.Match[str],
    phase_key: str,
    params_key: str = NO_PARAMS,
) -> dict[str, Any]:
    lang_key = match.group("lang")
    variant_key = match.group("variant")
    D, N, K = parse_config_match(match)

    return {
        "phase_key": phase_key,
        "phase": PHASE_MAP[phase_key],
        "variant_key": variant_key,
        "variant": variant_display_name(variant_key),
        "params_key": params_key,
        "params": params_display_name(params_key),
        "language_key": lang_key,
        "language": LANG_MAP[lang_key],
        "dimensions": D,
        "samples": N,
        "clusters": K,
        "config_id": format_config_id(D, N, K),
    }


def parse_benchmark_filename(path: Path) -> dict[str, Any] | None:
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


def parse_metrics_filename(path: Path, phase_key: str) -> dict[str, Any] | None:
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
