from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def convert_dataset_f32_to_f64(
    input_path: str | Path,
    output_path: str | Path,
    *,
    n_samples: int,
    n_features: int,
) -> None:
    source = np.memmap(
        Path(input_path).expanduser(),
        dtype=np.float32,
        mode="r",
        shape=(n_samples, n_features),
    )
    destination = np.ascontiguousarray(source, dtype=np.float64)
    if not np.all(np.isfinite(destination)):
        raise ValueError("HDBSCAN float64 dataset contains NaN or infinite values")

    ensure_parent_dir(output_path)
    destination.tofile(Path(output_path).expanduser())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the HDBSCAN benchmark input as a float64 binary matrix."
    )
    parser.add_argument("--dataset-bin", required=True)
    parser.add_argument("--hdbscan-dataset-out", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset_f32_to_f64(
        args.dataset_bin,
        args.hdbscan_dataset_out,
        n_samples=args.N,
        n_features=args.D,
    )


if __name__ == "__main__":
    main()
