"""Shared benchmark artifact keys and display labels."""

NO_PARAMS = "default"
REFERENCE_VARIANT = "reference"

LANGUAGE_CPP_KEY = "cpp"
LANGUAGE_PY_KEY = "py"

PHASE_DISPLAY_NAMES = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Algorithm",
    "gmm": "GaussianMixture EM",
}
PHASE_KEYS = tuple(PHASE_DISPLAY_NAMES)

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
    return PHASE_DISPLAY_NAMES.get(phase_key, phase_key)


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
