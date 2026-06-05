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


def format_config_id(D: int, N: int, K: int) -> str:
    return f"{D}D_{N}N_{K}K"


def format_configuration(D: int, N: int, K: int) -> str:
    return f"{D}D | {N}N | {K}K"


CONFIG_ID_PATTERN = r"(?P<D>\d+)D_(?P<N>\d+)N_(?P<K>\d+)K"

BENCHMARK_JSON_RE = re.compile(
    rf"^(?P<phase>soa|pp|lloyd|gmm)_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
LLOYD_METRICS_JSON_RE = re.compile(
    rf"^lloyd_metrics_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)
GMM_METRICS_JSON_RE = re.compile(
    rf"^gmm_metrics_(?P<lang>cpp|py)_{CONFIG_ID_PATTERN}\.json$"
)


def parse_config_match(match: re.Match[str]) -> tuple[int, int, int]:
    D = int(match.group("D"))
    N = int(match.group("N"))
    K = int(match.group("K"))
    return D, N, K


def parse_benchmark_filename(path: Path) -> dict[str, Any] | None:
    match = BENCHMARK_JSON_RE.match(path.name)
    if not match:
        return None

    phase_key = match.group("phase")
    lang_key = match.group("lang")
    D, N, K = parse_config_match(match)

    return {
        "phase_key": phase_key,
        "phase": PHASE_MAP[phase_key],
        "language_key": lang_key,
        "language": LANG_MAP[lang_key],
        "dimensions": D,
        "samples": N,
        "clusters": K,
        "config_id": format_config_id(D, N, K),
        "configuration": format_configuration(D, N, K),
    }
