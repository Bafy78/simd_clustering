import re
from pathlib import Path
from typing import Any


PHASE_MAP = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Iterations",
    "gmm": "GaussianMixture EM",
}

LANG_MAP = {
    "cpp": "C++",
    "py": "Python",
}


BENCHMARK_JSON_RE = re.compile(
    r"^(?P<phase>soa|pp|lloyd|gmm)_(?P<lang>cpp|py)_(?P<dim>\d+)D_(?P<samples>\d+)S_(?P<clusters>\d+)K\.json$"
)
LLOYD_PARITY_JSON_RE = re.compile(
    r"^lloyd_parity_(?P<dim>\d+)D_(?P<samples>\d+)S_(?P<clusters>\d+)K\.json$"
)
GMM_METRICS_JSON_RE = re.compile(
    r"^gmm_metrics_(?P<lang>cpp|py)_(?P<dim>\d+)D_(?P<samples>\d+)S_(?P<clusters>\d+)K\.json$"
)


def parse_benchmark_filename(path: Path) -> dict[str, Any] | None:
    match = BENCHMARK_JSON_RE.match(path.name)
    if not match:
        return None

    phase_key = match.group("phase")
    lang_key = match.group("lang")
    dim = int(match.group("dim"))
    samples = int(match.group("samples"))
    clusters = int(match.group("clusters"))

    return {
        "phase_key": phase_key,
        "phase": PHASE_MAP[phase_key],
        "language_key": lang_key,
        "language": LANG_MAP[lang_key],
        "dimensions": dim,
        "samples": samples,
        "clusters": clusters,
        "config_id": f"{dim}D_{samples}S_{clusters}K",
        "configuration": f"{dim}D | {samples}S | {clusters}K",
    }
