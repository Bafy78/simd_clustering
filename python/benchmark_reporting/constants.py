# Global Constants
LANG_CPP = "C++ (EVE)"
LANG_PY = "Python (Scikit-Learn)"

PHASE_MAP = {
    "soa": "AoS to SoA Tax",
    "pp": "K-Means++ Initialization",
    "lloyd": "Lloyd Iterations",
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
COL_PHASE_KEY = "Phase_Key"
COL_LANGUAGE = "Language"
COL_LANGUAGE_KEY = "Language_Key"

COL_DIMENSIONS = "D"
COL_SAMPLES = "N"
COL_CLUSTERS = "K"
COL_ITERATIONS = "Iterations"
COL_TIME_S = "Time_s"
COL_CONFIGURATION = "Configuration"
COL_CONFIG_ID = "Config_ID"

# Summary-time metadata
COL_TIME_FIELD = "Time_Field"
COL_TIME_STATISTIC = "Time_Statistic"
COL_PROCESS_COUNT = "Process_Count"
COL_TIMING_VALUE_COUNT = "Timing_Value_Count"
COL_INERTIA = "Inertia"
COL_COVARIANCE_TYPE = "Covariance_Type"
COL_CONVERGED = "Converged"
COL_LOWER_BOUND = "Lower_Bound"

# Derived / analysis columns
COL_SPEEDUP = "Speedup (x)"
COL_SPEEDUP_CI_LOW = "Speedup CI Low"
COL_SPEEDUP_CI_HIGH = "Speedup CI High"
COL_SPEEDUP_CI_LEVEL = "Speedup CI Level"
COL_SPEEDUP_STATISTIC = "Speedup Statistic"
COL_SPEEDUP_CI_LOWER_ERROR = "Speedup CI Lower Error"
COL_SPEEDUP_CI_UPPER_ERROR = "Speedup CI Upper Error"
COL_SPEEDUP_ERROR_WIDTH = "Speedup CI width"

COL_TIME_PER_ITER_PER_SAMPLE_MS = "Time per iteration per sample (ms)"
COL_TIME_PER_ITER_PER_SAMPLE_LOW_MS = "Time per iteration per sample low (ms)"
COL_TIME_PER_ITER_PER_SAMPLE_HIGH_MS = "Time per iteration per sample high (ms)"
COL_TIME_PER_ITER_PER_SAMPLE_LOWER_ERROR_MS = (
    "Time per iteration per sample lower error (ms)"
)
COL_TIME_PER_ITER_PER_SAMPLE_UPPER_ERROR_MS = (
    "Time per iteration per sample upper error (ms)"
)
COL_RUN_TO_RUN_SPREAD = "Run-to-run spread"

COL_CPP_POINT = "C++ Point"
COL_PY_POINT = "Python Point"

COL_TIME_PER_ITER = "Time_Per_Iter"
COL_EQUIVALENT_ITERS = "Equivalent_Iters"
COL_TIME_PER_ITER_MS = "Time_Per_Iter_ms"
COL_PY_MEAN_TIME = "Py_Mean_Time"
COL_BASE_SPEEDUP = "Base_Speedup"
COL_RETENTION = "Performance Retention (%)"
COL_ITERATION = "Iteration"

# Common column groups
CONFIG_COLS = [COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
CONFIG_LANGUAGE_COLS = CONFIG_COLS + [COL_LANGUAGE]
PHASE_CONFIG_COLS = [COL_PHASE] + CONFIG_COLS
PHASE_CONFIG_LANGUAGE_COLS = [COL_PHASE] + CONFIG_LANGUAGE_COLS
