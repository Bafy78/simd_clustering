import argparse
import copy
import json
import os
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def bench_name(bench: dict[str, Any]) -> str | None:
    return bench.get("metadata", {}).get("name")


def bench_unit(bench: dict[str, Any]) -> str | None:
    return bench.get("metadata", {}).get("unit")


def validate_compatible(
    base: dict[str, Any], other: dict[str, Any], other_path: str
) -> None:
    base_benchmarks = base.get("benchmarks", [])
    other_benchmarks = other.get("benchmarks", [])

    if len(base_benchmarks) != len(other_benchmarks):
        raise SystemExit(
            f"{other_path}: benchmark count mismatch: "
            f"{len(other_benchmarks)} != {len(base_benchmarks)}"
        )

    for i, (base_bench, other_bench) in enumerate(
        zip(base_benchmarks, other_benchmarks)
    ):
        base_name = bench_name(base_bench)
        other_name = bench_name(other_bench)

        if base_name and other_name and base_name != other_name:
            raise SystemExit(
                f"{other_path}: benchmark #{i} name mismatch: "
                f"{other_name!r} != {base_name!r}"
            )

        base_unit = bench_unit(base_bench)
        other_unit = bench_unit(other_bench)

        if base_unit and other_unit and base_unit != other_unit:
            raise SystemExit(
                f"{other_path}: benchmark #{i} unit mismatch: "
                f"{other_unit!r} != {base_unit!r}"
            )


def merge(paths: list[str]) -> dict[str, Any]:
    if not paths:
        raise SystemExit("No input files provided")

    loaded = [load_json(path) for path in paths]

    merged = copy.deepcopy(loaded[0])

    for bench in merged["benchmarks"]:
        bench["runs"] = []

    for timing_process_index, (path, suite) in enumerate(zip(paths, loaded)):
        validate_compatible(loaded[0], suite, path)

        for bench_index, bench in enumerate(suite["benchmarks"]):
            runs = bench.get("runs", [])

            if not runs:
                raise SystemExit(f"{path}: benchmark #{bench_index} has no runs")

            for run_index, run in enumerate(runs):
                run_copy = copy.deepcopy(run)
                metadata = run_copy.setdefault("metadata", {})
                metadata["timing_process_index"] = timing_process_index
                metadata["timing_process_json"] = os.path.basename(path)
                metadata["timing_process_pyperf_run_index"] = run_index

                merged["benchmarks"][bench_index]["runs"].append(run_copy)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    merged = merge(args.inputs)
    save_json(args.output, merged)


if __name__ == "__main__":
    main()
