# Global Constants
LANG_CPP = "C++ (EVE)"
LANG_PY = "Python (Scikit-Learn)"
PHASE_MAP = {
    "soa": "AoS to SoA",
    "pp": "K-Means++ Init",
    "lloyd": "Lloyd Algorithm"
}
# Column constants
COL_PHASE = "Phase"
COL_LANGUAGE = "Language"
COL_DIMENSIONS = "Dimensions"
COL_SAMPLES = "Samples"
COL_CLUSTERS = "Clusters"
COL_ITERATIONS = "Iterations"
COL_TIME_S = "Time_s"
COL_CONFIGURATION = "Configuration"

# Derived / analysis columns
COL_SPEEDUP = "Speedup (x)"
COL_TIME_PER_ITER = "Time_Per_Iter"
COL_EQUIVALENT_ITERS = "Equivalent_Iters"
COL_TIME_PER_ITER_MS = "Time_Per_Iter_ms"
COL_PY_MEAN_TIME = "Py_Mean_Time"
COL_BASE_SPEEDUP = "Base_Speedup"
COL_RETENTION = "Performance Retention (%)"
COL_ITERATION = "Iteration"
COL_NBR_CLUSTERS = "Nbr of Clusters"

# Common column groups
CONFIG_COLS = [COL_DIMENSIONS, COL_SAMPLES, COL_CLUSTERS]
CONFIG_LANGUAGE_COLS = CONFIG_COLS + [COL_LANGUAGE]
PHASE_CONFIG_COLS = [COL_PHASE] + CONFIG_COLS
PHASE_CONFIG_LANGUAGE_COLS = [COL_PHASE] + CONFIG_LANGUAGE_COLS