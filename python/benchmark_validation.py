from pathlib import Path
import pandas as pd

from .benchmark_io import load_lloyd_parity_summary


def validate_lloyd_parity_parallel(
    data_dir=Path("./datasets"),
    tolerance_pct=0.01,
):
    """
    Compatibility wrapper for the old notebook API.

    This no longer recomputes Lloyd parity from raw result files.
    It reads the parity block from benchmark_summary.json.
    """
    df_parity = load_lloyd_parity_summary(
        data_dir=data_dir,
        tolerance_pct=tolerance_pct,
    )

    if df_parity.empty:
        print("WARNING: No Lloyd parity records found in benchmark_summary.json.")
        return pd.DataFrame()

    failures = df_parity[df_parity["Status"] == "❌ FAIL"]

    if not failures.empty:
        print(
            f"WARNING: Found {len(failures)} configurations outside "
            f"{tolerance_pct}% limit!"
        )
    else:
        print(f"SUCCESS: All {len(df_parity)} configurations match within limit!")

    return df_parity
