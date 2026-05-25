import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmark_pipeline.paths import BIN_DIR, REPO_ROOT, repo_path
from benchmark_pipeline.runner import run_command
from benchmark_pipeline.tasks import build_pipeline, config_id, dataset_path

PROFILE_EVENTS = "Ir,Dr,D1mr,DLmr,Dw,D1mw,DLmw"


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        print(f"Required tool not found on PATH: {tool}")
        sys.exit(1)


def compile_profile_binary(dim: int) -> Path:
    os.makedirs(BIN_DIR, exist_ok=True)

    src = repo_path("cpp", "benchmarks", "profile_lloyd_callgrind.cpp")
    out = BIN_DIR / f"profile_lloyd_callgrind_{dim}D.bin"

    cmd = [
        "g++-14",
        "-O3",
        "-g",
        "-march=native",
        "-std=c++20",
        "-I../eve/include",
        "-I../nanobench/src/include",
        f"-DTUPLE_SIZE={dim}",
        "-DKMEANS_K_TILE=5",
        "-DKMEANS_M_GROUP=2",
        src,
        "-o",
        str(out),
    ]

    print(f"Compiling Callgrind Lloyd binary for {dim}D...")
    run_command("Compile Callgrind Lloyd binary", cmd)
    return out


def run_and_capture(
    name: str, command: list[str], stdout_path: Path, stderr_path: Path
) -> None:
    print(f"[{name}] {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if result.returncode != 0:
        print(f"\n{name} FAILED")
        print(f"stdout: {stdout_path}")
        print(f"stderr: {stderr_path}")
        print(result.stderr)
        sys.exit(result.returncode)


def add_cache_options(command: list[str], args: argparse.Namespace) -> None:
    if args.I1:
        command.append(f"--I1={args.I1}")
    if args.D1:
        command.append(f"--D1={args.D1}")
    if args.LL:
        command.append(f"--LL={args.LL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one Lloyd configuration under Callgrind with cache simulation enabled."
    )
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--out-dir", default=repo_path("callgrind_results"))

    # Optional explicit cache model. Example:
    # --D1 32768,8,64 --LL 33554432,16,64
    parser.add_argument("--I1")
    parser.add_argument("--D1")
    parser.add_argument("--LL")

    args = parser.parse_args()

    require_tool("valgrind")
    require_tool("callgrind_annotate")

    case = config_id(args.dim, args.n_samples, args.n_clusters)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = BIN_DIR / f"profile_lloyd_callgrind_{args.dim}D.bin"

    if not args.skip_compile:
        binary = compile_profile_binary(args.dim)

    if not args.skip_generate:
        dataset_task = build_pipeline(
            args.dim,
            args.n_samples,
            args.n_clusters,
            bench_processes=1,
            bench_values=1,
            bench_min_time=0.0,
        )[0]
        run_command(dataset_task.name, dataset_task.command)

    dataset_bin = dataset_path(f"data_{case}.bin")
    init_bin = dataset_path(f"init_{case}.bin")

    callgrind_out = out_dir / f"callgrind.lloyd.{case}.out"
    metrics_out = out_dir / f"lloyd_metrics_cpp.{case}.json"
    valgrind_stdout = out_dir / f"valgrind.callgrind.lloyd.{case}.stdout.txt"
    valgrind_stderr = out_dir / f"valgrind.callgrind.lloyd.{case}.stderr.txt"
    annotate_out = out_dir / f"callgrind.lloyd.{case}.annotate.txt"
    annotate_err = out_dir / f"callgrind.lloyd.{case}.annotate.stderr.txt"

    callgrind_cmd = [
        "valgrind",
        "--tool=callgrind",
        "--cache-sim=yes",
        "--branch-sim=no",
        "--instr-atstart=no",
        f"--callgrind-out-file={callgrind_out}",
    ]
    add_cache_options(callgrind_cmd, args)
    callgrind_cmd.extend(
        [
            str(binary),
            dataset_bin,
            str(args.n_samples),
            str(args.n_clusters),
            init_bin,
            str(metrics_out),
        ]
    )

    run_and_capture(
        "Callgrind Lloyd",
        callgrind_cmd,
        valgrind_stdout,
        valgrind_stderr,
    )

    callgrind_annotate_cmd = [
        "callgrind_annotate",
        f"--show={PROFILE_EVENTS}",
        "--sort=D1mr,D1mw,DLmr,DLmw,Ir",
        "--inclusive=yes",
        "--tree=both",
        "--threshold=0.0",
        "--context=8",
        str(callgrind_out),
    ]

    run_and_capture(
        "callgrind_annotate",
        callgrind_annotate_cmd,
        annotate_out,
        annotate_err,
    )

    print("\nDone.")
    print(f"Callgrind raw output:      {callgrind_out}")
    print(f"Callgrind annotated:       {annotate_out}")
    print(f"Metrics JSON:              {metrics_out}")
    print(f"Valgrind stdout:           {valgrind_stdout}")
    print(f"Valgrind stderr:           {valgrind_stderr}")
    print("")
    print("Open the raw output directly in KCachegrind, for example:")
    print(f"  kcachegrind {callgrind_out}")


if __name__ == "__main__":
    main()
