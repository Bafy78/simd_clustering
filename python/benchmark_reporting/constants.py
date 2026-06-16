# Global Constants
LANG_CPP = "C++ (EVE)"
LANG_PY = "Python (Scikit-Learn)"

PHASE_MAP = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Algorithm",
    "gmm": "GaussianMixture EM",
}

PHASE_ORDER = list(PHASE_MAP.values())

LANG_MAP = {
    "cpp": LANG_CPP,
    "py": LANG_PY,
}

LANG_ORDER = [LANG_CPP, LANG_PY]

# Column constants
COL_PHASE = "Phase"
COL_LANGUAGE = "Language"
COL_VARIANT = "Variant"
COL_PARAMS = "Params"

COL_DIMENSIONS = "D"
COL_SAMPLES = "N"
COL_CLUSTERS = "K"
COL_ALGORITHM_ITERATIONS = "Algorithm_Iterations"
COL_TIME_S = "Time_s"

# Summary-time metadata
COL_TIME_FIELD = "Time_Field"
COL_TIME_STATISTIC = "Time_Statistic"
COL_TIMING_PROCESS_COUNT = "Timing_Process_Count"
COL_TIMING_VALUE_COUNT = "Timing_Value_Count"
COL_INERTIA = "Inertia"
COL_COVARIANCE_TYPE = "Covariance_Type"
COL_LOWER_BOUND = "Lower_Bound"

COL_EXCLUSION_REASON = "Exclusion Reason"
COL_EXCLUSION_RULES = "Exclusion Rules"

# Derived / analysis columns
COL_SPEEDUP = "Speedup (x)"
COL_SPEEDUP_CI_LOW = "Speedup CI Low"
COL_SPEEDUP_CI_HIGH = "Speedup CI High"
COL_SPEEDUP_CI_LEVEL = "Speedup CI Level"
COL_SPEEDUP_STATISTIC = "Speedup Statistic"
COL_SPEEDUP_CI_LOWER_ERROR = "Speedup CI Lower Error"
COL_SPEEDUP_CI_UPPER_ERROR = "Speedup CI Upper Error"
COL_SPEEDUP_ERROR_WIDTH = "Speedup CI width"

COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_MS = (
    "Time per algorithm iteration per sample (ms)"
)
COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_LOW_MS = (
    "Time per algorithm iteration per sample low (ms)"
)
COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_HIGH_MS = (
    "Time per algorithm iteration per sample high (ms)"
)
COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_LOWER_ERROR_MS = (
    "Time per algorithm iteration per sample lower error (ms)"
)
COL_TIME_PER_ALGORITHM_ITER_PER_SAMPLE_UPPER_ERROR_MS = (
    "Time per algorithm iteration per sample upper error (ms)"
)
COL_TIMING_RUN_SPREAD = "Timing run spread"

COL_CPP_POINT = "C++ Point"
COL_PY_POINT = "Python Point"

COL_TIME_PER_ALGORITHM_ITER = "Time_Per_Algorithm_Iter"
COL_EQUIVALENT_ALGORITHM_ITERS = "Equivalent_Algorithm_Iters"
COL_TIME_PER_ALGORITHM_ITER_MS = "Time_Per_Algorithm_Iter_ms"
COL_PY_MEAN_TIME = "Py_Mean_Time"
COL_BASE_SPEEDUP = "Base_Speedup"
COL_RETENTION = "Performance Retention (%)"
COL_ALGORITHM_ITERATION = "Algorithm Iteration"

# Common column groups
CONFIG_COLS = [COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
CONFIG_LANGUAGE_COLS = CONFIG_COLS + [COL_LANGUAGE]
CONFIG_VARIANT_COLS = CONFIG_COLS + [COL_VARIANT]
CONFIG_VARIANT_PARAMS_COLS = CONFIG_COLS + [COL_VARIANT, COL_PARAMS]
PHASE_CONFIG_COLS = [COL_PHASE] + CONFIG_COLS
PHASE_CONFIG_VARIANT_COLS = [COL_PHASE, COL_VARIANT] + CONFIG_COLS
PHASE_CONFIG_VARIANT_PARAMS_COLS = [COL_PHASE, COL_VARIANT, COL_PARAMS] + CONFIG_COLS
PHASE_CONFIG_LANGUAGE_COLS = [COL_PHASE] + CONFIG_LANGUAGE_COLS
PHASE_CONFIG_VARIANT_LANGUAGE_COLS = [COL_PHASE, COL_VARIANT] + CONFIG_LANGUAGE_COLS
PHASE_CONFIG_VARIANT_PARAMS_LANGUAGE_COLS = [COL_PHASE, COL_VARIANT, COL_PARAMS] + CONFIG_LANGUAGE_COLS
