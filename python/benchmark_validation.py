from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd

from .benchmark_constants import *
from .benchmark_io import read_result_iterations_and_inertia, extract_config_params

def _validate_one_config(args):
    cpp_file, py_file, data_file, config_id, tolerance_pct = args

    dim, samples, clusters = extract_config_params(config_id)

    raw_data = np.fromfile(
        data_file,
        dtype=np.float32,
        count=samples * dim,
    ).reshape(samples, dim)

    cpp_iters, cpp_inertia = read_result_iterations_and_inertia(cpp_file, raw_data)
    py_iters, py_inertia = read_result_iterations_and_inertia(py_file, raw_data)

    max_inertia = max(cpp_inertia, py_inertia)
    diff_pct = (
        abs(cpp_inertia - py_inertia) / max_inertia * 100
        if max_inertia > 0
        else 0.0
    )

    return {
        COL_CONFIGURATION: f"{dim}D | {samples}S | {clusters}K",
        "Diff (%)": diff_pct,
        "Status": "✅ PASS" if diff_pct <= tolerance_pct else "❌ FAIL",
        "Lloyd C++ Iteration": cpp_iters,
        "Lloyd Py Iterations": py_iters,
        "C++ Inertia": cpp_inertia,
        "Py Inertia": py_inertia,
    }


def validate_lloyd_parity_parallel(
    data_dir=Path("./datasets"),
    tolerance_pct=0.01,
    max_workers=None,
):
    path = Path(data_dir)

    jobs = []
    for cpp_file in path.glob("results_cpp_*.txt"):
        config_id = cpp_file.name.replace("results_cpp_", "").replace(".txt", "")
        py_file = path / f"results_py_{config_id}.txt"
        data_file = path / f"data_{config_id}.bin"

        if not py_file.exists() or not data_file.exists():
            continue

        jobs.append((cpp_file, py_file, data_file, config_id, tolerance_pct))

    if not jobs:
        print("WARNING: No matching configurations found.")
        return pd.DataFrame()

    if max_workers is None:
        # Do not blindly use every core; that can saturate disk I/O.
        max_workers = min(4, os.cpu_count() or 1, len(jobs))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(_validate_one_config, jobs))

    df_parity = pd.DataFrame(records)

    failures = df_parity[df_parity["Status"] == "❌ FAIL"]

    if not failures.empty:
        print(f"WARNING: Found {len(failures)} configurations outside {tolerance_pct}% limit!")
    else:
        print(f"SUCCESS: All {len(df_parity)} configurations match within limit!")

    return df_parity.sort_values(by="Diff (%)", ascending=False).reset_index(drop=True)