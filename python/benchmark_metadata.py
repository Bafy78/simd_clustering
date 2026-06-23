"""Shared benchmark artifact keys and display labels."""

NO_PARAMS = "default"
REFERENCE_VARIANT = "reference"

LANGUAGE_CPP_KEY = "cpp"
LANGUAGE_PY_KEY = "py"

FULL_STAGE_KEY = "full"

HDBSCAN_DISTANCE_STAGE_KEY = "distance"
HDBSCAN_CORE_STAGE_KEY = "core"
HDBSCAN_MREACH_STAGE_KEY = "mreach"
HDBSCAN_MST_STAGE_KEY = "mst"
HDBSCAN_LINKAGE_STAGE_KEY = "linkage"
HDBSCAN_SELECT_STAGE_KEY = "select"
HDBSCAN_STAGE_KEYS = (
    HDBSCAN_DISTANCE_STAGE_KEY,
    HDBSCAN_CORE_STAGE_KEY,
    HDBSCAN_MREACH_STAGE_KEY,
    HDBSCAN_MST_STAGE_KEY,
    HDBSCAN_LINKAGE_STAGE_KEY,
    HDBSCAN_SELECT_STAGE_KEY,
    FULL_STAGE_KEY,
)

PHASE_DISPLAY_NAMES = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Algorithm",
    "gmm": "GaussianMixture EM",
    "hdbscan": "HDBSCAN",
}
PHASE_KEYS = tuple(PHASE_DISPLAY_NAMES)

STAGE_DISPLAY_NAMES = {
    FULL_STAGE_KEY: "Full",
    HDBSCAN_DISTANCE_STAGE_KEY: "Distance Matrix",
    HDBSCAN_CORE_STAGE_KEY: "Core Distances",
    HDBSCAN_MREACH_STAGE_KEY: "Mutual Reachability",
    HDBSCAN_MST_STAGE_KEY: "Minimum Spanning Tree",
    HDBSCAN_LINKAGE_STAGE_KEY: "Single Linkage Tree",
    HDBSCAN_SELECT_STAGE_KEY: "Cluster Selection",
}

PHASE_STAGE_KEYS = {
    phase_key: (FULL_STAGE_KEY,)
    for phase_key in PHASE_KEYS
}
PHASE_STAGE_KEYS["hdbscan"] = HDBSCAN_STAGE_KEYS

LANGUAGE_DISPLAY_NAMES = {
    LANGUAGE_CPP_KEY: "C++",
    LANGUAGE_PY_KEY: "Python",
}

REPORTING_LANGUAGE_DISPLAY_NAMES = {
    LANGUAGE_CPP_KEY: "C++ (EVE)",
    LANGUAGE_PY_KEY: "Python (Scikit-Learn)",
}

VARIANT_DISPLAY_NAMES = {
    "static": "Static",
    "dynamic": "Dynamic",
    "auto": "Auto",
    REFERENCE_VARIANT: "Reference",
}


def format_config_id(D: int, N: int, K: int) -> str:
    return f"{D}D_{N}N_{K}K"


def display_name(key: str) -> str:
    return key.replace("_", " ").title()


def phase_display_name(phase_key: str) -> str:
    return PHASE_DISPLAY_NAMES[phase_key]


def fallback_phase_display_name(phase_key: str) -> str:
    return PHASE_DISPLAY_NAMES.get(phase_key, display_name(phase_key))


def stage_display_name(stage_key: str) -> str:
    return STAGE_DISPLAY_NAMES.get(stage_key, display_name(stage_key))


def phase_stage_keys(phase_key: str) -> tuple[str, ...]:
    return PHASE_STAGE_KEYS.get(phase_key, (FULL_STAGE_KEY,))


def all_stage_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for stage_keys in PHASE_STAGE_KEYS.values():
        for stage_key in stage_keys:
            if stage_key not in keys:
                keys.append(stage_key)
    return tuple(keys)


def language_display_name(language_key: str) -> str:
    return LANGUAGE_DISPLAY_NAMES[language_key]


def reporting_language_display_name(language_key: str) -> str:
    return REPORTING_LANGUAGE_DISPLAY_NAMES[language_key]


def variant_display_name(variant_key: str) -> str:
    return VARIANT_DISPLAY_NAMES.get(variant_key, display_name(variant_key))


def params_display_name(params_key: str) -> str:
    if params_key == NO_PARAMS:
        return "Default"
    return display_name(params_key)
